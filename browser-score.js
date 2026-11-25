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

function checkedSentences(text, sentences) {
  if (!Array.isArray(sentences) || !sentences.length || sentences.length > text.length) {
    throw new Error('句子列表无效。');
  }
  let cursor = 0;
  const pieces = [];
  for (const sentence of sentences) {
    if (typeof sentence !== 'string') throw new Error('句子格式无效。');
    const piece = sentence.trim();
    while (cursor < text.length && /\s/u.test(text[cursor])) cursor++;
    if (!piece || !text.startsWith(piece, cursor)) throw new Error('句子顺序与原文不符。');
    pieces.push(piece);
    cursor += piece.length;
  }
  if (text.slice(cursor).trim()) throw new Error('句子列表未覆盖全部原文。');
  return pieces;
}

// The worker retains a bounded set of exact inputs for rechecks and small edits.
// Nothing from the essay is written to browser storage.
const recent = new Map();
let savedCharacters = 0;

function remember(text, entry) {
  if (recent.delete(text)) savedCharacters -= text.length;
  recent.set(text, entry);
  savedCharacters += text.length;
  while (recent.size > 512 || savedCharacters > 240000) {
    const oldest = recent.keys().next().value;
    recent.delete(oldest);
    savedCharacters -= oldest.length;
  }
}

function validateText(text) {
  if (typeof text !== 'string' || !/[A-Za-z]/.test(text) || text.length > 60000) {
    throw new Error('请输入英文文本，最多 60,000 个字符。');
  }
}

async function scoreTexts(distinct, notify, completed, batchSentences = false) {
  const byText = new Map();
  const pending = [];
  for (const text of distinct) {
    const cached = recent.get(text);
    if (cached && (batchSentences || !cached.usedSentenceGraph)) {
      remember(text, cached);
      byText.set(text, cached.result);
      completed?.(text, cached.result);
    } else {
      pending.push(text);
    }
  }
  if (!pending.length) return byText;
  if (!loading) loading = loadModel(notify).catch(error => { loading = undefined; throw error; });
  const {handle, tokenizer} = await loading;
  const {classifyMany} = await import('./score-staged.js?v=6');
  const borderline = [];
  function record(text, result, usedSentenceGraph) {
    remember(text, {result, usedSentenceGraph});
    byText.set(text, result);
    completed?.(text, result);
  }
  await classifyMany(pending, tokenizer, handle, notify, (index, result, usedSentenceGraph) => {
    const text = pending[index];
    // Recheck decisions near the threshold using the original scalar graph.
    if (usedSentenceGraph && Math.abs(result.score - threshold) < 0.0001) borderline.push(text);
    else record(text, result, usedSentenceGraph);
  }, batchSentences);
  if (borderline.length) {
    await classifyMany(borderline, tokenizer, handle, notify, (index, result) => {
      record(borderline[index], result, false);
    });
  }
  return byText;
}

export async function scoreWhole(text, notify = () => {}) {
  validateText(text);
  return (await scoreTexts([text], notify)).get(text);
}

export async function scoreEssay(text, notify = () => {}, sentences, onSentence) {
  validateText(text);
  if (sentences !== undefined) {
    const parts = checkedSentences(text, sentences);
    const whole = await scoreWhole(text, notify);
    const locations = new Map();
    for (const [index, part] of parts.entries()) {
      if (!locations.has(part)) locations.set(part, []);
      locations.get(part).push(index);
    }
    const rows = new Array(parts.length);
    await scoreTexts(locations.keys(), notify, (part, result) => {
      for (const index of locations.get(part)) {
        const row = {text: part, score: result.score};
        rows[index] = row;
        onSentence?.(index, row);
      }
    }, true);
    return {...whole, sentences: rows};
  }

  const parts = text.split(/\r\n|[\r\n\v\f\u001c-\u001e\u0085\u2028\u2029]/u)
    .map(line => line.trim()).filter(Boolean);
  const wholeKey = text.trim();
  const distinct = [text, ...new Set(parts.filter(part => part !== wholeKey))];
  const results = await scoreTexts(distinct, notify);
  const whole = results.get(text);
  const paragraphs = parts.map(line => ({
    text: line,
    words: line.split(/\s+/u).length,
    score: (line === wholeKey ? whole : results.get(line)).score,
  }));
  const total_words = paragraphs.reduce((total,p)=>total+p.words,0);
  const flagged_words = paragraphs.reduce((total,p)=>total+(p.score>=threshold?p.words:0),0);
  return {...whole,paragraph_analysis:{
    paragraphs,total_words,flagged_words,unknown_words:0,flagged_fraction:flagged_words/total_words,
    unknown_fraction:0,coverage:1,threshold,min_words:1,
  }};
}
