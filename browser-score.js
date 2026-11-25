// Runs inside a module worker. Only public model assets leave the page's origin.
import {
  AutoTokenizer, env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js';

const repository = 'GabeuxDev/ai-text-detector-v1.01-onnx';
const revision = '1017cdcc0fa315991a32c5a0eb7e1f8c5e74bf34';
const threshold = 0.98;

env.allowLocalModels = false;
let loading;

const weightBytes = 1736052736;
const weightFile = 'model.onnx_data';
const weightURL = 'https://huggingface.co/' + repository + '/resolve/' + revision + '/onnx/' + weightFile;

// Stream the external weights into private browser storage. This avoids creating
// an additional 1.74 GB Response/Blob copy to populate the default model cache.
async function loadWeights(notify) {
  if (!navigator.storage?.getDirectory) throw new Error('浏览器不支持私有文件缓存。请用新版桌面 Edge 或 Chrome 打开。');
  const root = await navigator.storage.getDirectory();
  const handle = await root.getFileHandle('essaylens-' + revision + '.weights', {create: true});
  if ((await handle.getFile()).size === weightBytes) {
    notify('已找到本机缓存的模型权重。');
    return handle;
  }
  const response = await fetch(weightURL);
  if (!response.ok || !response.body) throw new Error('模型权重下载失败（HTTP ' + response.status + '）。');
  const reader = response.body.getReader();
  const writer = await handle.createWritable();
  const buffer = new Uint8Array(4 * 1024 * 1024);
  let filled = 0, loaded = 0, lastUpdate = 0;
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      let pos = 0;
      while (pos < value.length) {
        const count = Math.min(buffer.length - filled, value.length - pos);
        buffer.set(value.subarray(pos, pos + count), filled);
        filled += count;
        pos += count;
        if (filled === buffer.length) {
          await writer.write(buffer);
          filled = 0;
        }
      }
      loaded += value.length;
      if (Date.now() - lastUpdate > 400) {
        lastUpdate = Date.now();
        notify('正在下载权重 · ' + Math.floor(loaded / 1000000) + ' / 1737 MB');
      }
    }
    if (loaded !== weightBytes) throw new Error('权重下载长度异常，请重试。');
    if (filled) await writer.write(buffer.subarray(0, filled));
    await writer.close();
  } catch (error) {
    await writer.abort().catch(()=>{});
    await reader.cancel().catch(()=>{});
    throw error;
  }
  return handle;
}

async function loadModel(notify) {
  notify('正在准备英文分词器和模型权重，首次下载约 1.74 GB。');
  const tokenizer = await AutoTokenizer.from_pretrained(repository, {revision});
  const handle = await loadWeights(notify);
  return {handle, tokenizer};
}

export async function scoreEssay(text, notify = () => {}) {
  if (typeof text !== 'string' || !/[A-Za-z]/.test(text) || text.length > 60000) {
    throw new Error('请输入英文文本，最多 60,000 个字符。');
  }
  if (!loading) loading = loadModel(notify).catch(error => { loading = undefined; throw error; });
  const {handle, tokenizer} = await loading;
  const lines = text.split(/\r\n|[\r\n\v\f\u001c-\u001e\u0085\u2028\u2029]/u).map(line=>line.trim()).filter(Boolean);
  const distinct = [text, ...new Set(lines.filter(line=>line!==text.trim()))];
  const {classifyMany} = await import('./score-staged.js');
  const results = await classifyMany(distinct,tokenizer,handle,notify);
  const byText = new Map(distinct.map((line,i)=>[line.trim(),results[i]]));
  const paragraphs = lines.map(line=>({text:line,words:line.split(/\s+/u).length,score:byText.get(line).score}));
  const total_words = paragraphs.reduce((total,p)=>total+p.words,0);
  const flagged_words = paragraphs.reduce((total,p)=>total+(p.score>=threshold?p.words:0),0);
  return {...results[0],paragraph_analysis:{
    paragraphs,total_words,flagged_words,unknown_words:0,flagged_fraction:flagged_words/total_words,
    unknown_fraction:0,coverage:1,threshold,min_words:1,
  }};
}
