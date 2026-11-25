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
  const model = await AutoModelForSequenceClassification.from_pretrained(repository, {
    ...options, dtype: 'fp32', device: 'webgpu', use_external_data_format: true,
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
