// Runs inside a module worker. Only public model assets leave the page's origin.
import {
  AutoTokenizer, AutoModelForSequenceClassification, Tensor, env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js';

const repository = 'GabeuxDev/ai-text-detector-v1.01-onnx';
const revision = '1017cdcc0fa315991a32c5a0eb7e1f8c5e74bf34';
const threshold = 0.98;
const chunkTokens = 766;
env.allowLocalModels = false;
let loading;

const weightBytes = 1736052736;
const weightFile = 'model.onnx_data';
const weightURL = 'https://huggingface.co/' + repository + '/resolve/' + revision + '/onnx/' + weightFile;

// Stream the external weights into private browser storage. This avoids creating
// an additional 1.74 GB Response/Blob copy to populate the default model cache.
async function loadWeights(notify) {
  const root = await navigator.storage.getDirectory();
  const handle = await root.getFileHandle('essaylens-' + revision + '.weights', {create: true});
  let file = await handle.getFile();
  if (file.size !== weightBytes) {
    const response = await fetch(weightURL);
    if (!response.ok || !response.body) throw new Error('模型权重下载失败（HTTP ' + response.status + '）。');
    const writer = await handle.createWritable();
    let loaded = 0;
    let lastUpdate = 0;
    const tracker = new TransformStream({
      transform(chunk, controller) {
        loaded += chunk.byteLength;
        if (Date.now() - lastUpdate > 400) {
          lastUpdate = Date.now();
          notify('正在下载权重 · ' + Math.floor(loaded / 1000000) + ' / 1737 MB');
        }
        controller.enqueue(chunk);
      },
      flush() {
        if (loaded !== weightBytes) throw new Error('权重下载长度异常，请重试。');
      },
    });
    await response.body.pipeThrough(tracker).pipeTo(writer);
    file = await handle.getFile();
    if (file.size !== weightBytes) throw new Error('本机模型缓存写入失败。');
  }
  notify('正在从浏览器缓存读取模型权重…');
  const weights = new Uint8Array(weightBytes);
  const reader = file.stream().getReader();
  let offset = 0;
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    weights.set(value, offset);
    offset += value.byteLength;
  }
  if (offset !== weightBytes) throw new Error('本机模型缓存读取失败。');
  return weights;
}

async function loadModel(notify) {
  if (!navigator.gpu || !await navigator.gpu.requestAdapter()) {
    throw new Error('这个模型需要 WebGPU。请使用启用了硬件加速的桌面版 Microsoft Edge 或 Chrome。');
  }
  notify('正在加载模型和分词器。首次使用需要下载约 1.74 GB。');
  let lastUpdate = 0;
  const options = {
    revision,
    progress_callback(event) {
      if (event.status !== 'progress' || !event.loaded || Date.now() - lastUpdate < 400) return;
      lastUpdate = Date.now();
      const mb = Math.floor(event.loaded / 1000000);
      const total = event.total ? ' / ' + Math.ceil(event.total / 1000000) : '';
      notify('加载 ' + event.file.split('/').pop() + ' · ' + mb + total + ' MB');
    },
  };
  const tokenizer = await AutoTokenizer.from_pretrained(repository, options);
  const weights = await loadWeights(notify);
  notify('正在初始化 WebGPU 模型，首次运行可能需要几分钟…');
  const model = await AutoModelForSequenceClassification.from_pretrained(repository, {
    revision, dtype: 'fp32', device: 'webgpu', use_external_data_format: false,
    session_options: {externalData: [{path: weightFile, data: weights}]},
  });
  return {tokenizer, model};
}

async function classify(text, tokenizer, model) {
  const tokens = tokenizer(text, {add_special_tokens: false, return_tensor: false}).input_ids;
  if (tokens.length === 0) throw new Error('这段文字无法分词。');
  const chunks = Math.ceil(tokens.length / chunkTokens);
  const width = Math.ceil(tokens.length / chunks);
  let sum = 0;
  for (let start = 0; start < tokens.length; start += width) {
    const part = tokens.slice(start, start + width);
    const ids = [tokenizer.bos_token_id, ...part, tokenizer.sep_token_id];
    const dims = [1, ids.length];
    const input_ids = new Tensor('int64', BigInt64Array.from(ids, BigInt), dims);
    const attention_mask = new Tensor('int64', new BigInt64Array(ids.length).fill(1n), dims);
    const {logits} = await model({input_ids, attention_mask});
    const logit = Number(logits.data[0]);
    logits.dispose();
    if (!Number.isFinite(logit)) throw new Error('此设备未能计算有效的模型分数。');
    sum += part.length / (1 + Math.exp(-logit));
  }
  return {score: sum / tokens.length, chunks};
}

export async function scoreEssay(text, notify = () => {}) {
  if (typeof text !== 'string' || !/[A-Za-z]/.test(text) || text.length > 60000) {
    throw new Error('请输入英文文本，最多 60,000 个字符。');
  }
  if (!loading) loading = loadModel(notify).catch(error => { loading = undefined; throw error; });
  const {model, tokenizer} = await loading;
  notify('正在分析全文…');
  const whole = await classify(text, tokenizer, model);
  const cache = new Map([[text.trim(), whole.score]]);
  const lines = text.split(/\r\n|[\r\n\v\f\u001c-\u001e\u0085\u2028\u2029]/u).map(line => line.trim()).filter(Boolean);
  const paragraphs = [];
  let total_words = 0;
  let flagged_words = 0;
  for (const line of lines) {
    if (!cache.has(line)) {
      notify('正在分析段落 ' + (paragraphs.length + 1) + ' / ' + lines.length);
      cache.set(line, (await classify(line, tokenizer, model)).score);
    }
    const words = line.split(/\s+/u).length;
    const score = cache.get(line);
    total_words += words;
    if (score >= threshold) flagged_words += words;
    paragraphs.push({text: line, words, score});
  }
  return {...whole, paragraph_analysis: {
    paragraphs, total_words, flagged_words, unknown_words: 0,
    flagged_fraction: flagged_words / total_words, unknown_fraction: 0, coverage: 1,
    threshold, min_words: 1,
  }};
}
