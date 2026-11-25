// Low-memory fp32 inference. The four groups preserve the ONNX layer order.
// A read-only OPFS sync handle provides just the weights needed by each group.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.wasm.min.mjs';

const singleThread = globalThis.location && new URL(globalThis.location.href).searchParams.get('threads') === '1';
ort.env.wasm.numThreads = !singleThread && globalThis.crossOriginIsolated && typeof SharedArrayBuffer === 'function' ? 0 : 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
const graphDir = new URL('./stages/', import.meta.url);
const layouts = new Map();
function loadLayout(name) {
  if (!layouts.has(name)) {
    const pending = fetch(new URL(name, graphDir)).then(response => {
      if (!response.ok) throw Error('无法加载模型结构。');
      return response.json();
    }).catch(error => {
      layouts.delete(name);
      throw error;
    });
    layouts.set(name, pending);
  }
  return layouts.get(name);
}

// Tokenization can stay unchanged after a formatting edit, and a local edit in a
// long essay often leaves entire chunks intact. Only cache the original graph's
// scalar outputs, keyed by exact token IDs, for the lifetime of this worker.
const chunkCache = new Map();
function rememberChunk(key, denominator) {
  chunkCache.delete(key);
  chunkCache.set(key, denominator);
  if (chunkCache.size > 64) chunkCache.delete(chunkCache.keys().next().value);
}

function readBytes(handle, target, offset) {
  let n = 0;
  while (n < target.byteLength) {
    const count = handle.read(target.subarray(n), {at: offset + n});
    if (!count) throw new Error('缓存文件不完整。');
    n += count;
  }
}

// Group nearby lengths to share each layer's fixed positional projections and
// weight reads. A zero attention mask excludes padding throughout the graph,
// including the final mean pooling. Long chunks retain the original one-row path.
function makeBatches(jobs, allowBatching) {
  const ordered = allowBatching ? jobs.slice().sort((a, b) => a.tokens.length - b.tokens.length) : jobs;
  const batches = [];
  for (const job of ordered) {
    const width = job.tokens.length;
    const previous = batches[batches.length - 1];
    if (allowBatching && previous && previous.jobs.length < 6 && width <= 96 &&
        width <= previous.jobs[0].tokens.length * 1.5) {
      previous.jobs.push(job);
      previous.width = width;
    } else {
      batches.push({jobs: [job], width, scope: {}});
    }
  }
  return batches;
}

function addEmbeddings(batch, embedding, fh) {
  const {jobs, width, scope} = batch;
  const data = new Float32Array(jobs.length * width * 1024);
  const bytes = new Uint8Array(data.buffer);
  const mask = new BigInt64Array(jobs.length * width);
  for (const [row, job] of jobs.entries()) {
    for (const [column, id] of job.tokens.entries()) {
      const position = row * width + column;
      readBytes(fh, bytes.subarray(position * 4096, (position + 1) * 4096), id * 4096);
      mask[position] = 1n;
    }
  }
  scope[embedding] = new ort.Tensor('float32', data, [jobs.length, width, 1024]);
  scope.attention_mask = new ort.Tensor('int64', mask, [jobs.length, width]);
}

export async function classifyMany(texts, tokenizer, fileHandle, notify, onResult, batchSentences = false) {
  const jobs = [];
  const results = texts.map(() => ({score: 0, chunks: 0}));
  const remaining = [];
  const contributions = [];
  function finish(index, usedSentenceGraph) {
    const result = results[index];
    result.score = contributions[index].reduce((sum, value) => sum + value, 0);
    onResult?.(index, result, usedSentenceGraph);
  }
  for (const [index, text] of texts.entries()) {
    const ids = tokenizer(text, {add_special_tokens:false, return_tensor:false}).input_ids;
    if (!ids.length) throw new Error('这段文字无法分词。');
    const chunks = Math.ceil(ids.length / 766);
    results[index].chunks = chunks;
    remaining[index] = 0;
    contributions[index] = [];
    const width = Math.ceil(ids.length / chunks);
    for (let start = 0; start < ids.length; start += width) {
      const part = ids.slice(start, start + width);
      const key = part.join(',');
      const chunk = Math.floor(start / width);
      const weight = part.length / ids.length;
      if (chunkCache.has(key)) {
        const denominator = chunkCache.get(key);
        rememberChunk(key, denominator);
        contributions[index][chunk] = weight / denominator;
        continue;
      }
      const tokens = [tokenizer.bos_token_id, ...part, tokenizer.sep_token_id];
      jobs.push({index, chunk, key, weight, tokens});
      remaining[index]++;
    }
    if (!remaining[index]) finish(index, false);
  }
  if (!jobs.length) return results;
  const {embedding, stages} = await loadLayout(batchSentences ? 'layout-v6.json' : 'layout.json');
  const batches = makeBatches(jobs, batchSentences);
  const fh = await fileHandle.createSyncAccessHandle({mode: 'read-only'});
  const futureInputs = stages.map((_, index) => new Set(stages.slice(index + 1).flatMap(stage => stage.inputs)));
  try {
    let begin = 0;
    while (begin < batches.length) {
      let end = begin, tokensInWindow = 0;
      while (end < batches.length) {
        const batch = batches[end];
        const size = batch.width * batch.jobs.length;
        if (tokensInWindow + size > 1536) break;
        tokensInWindow += size;
        end++;
      }
      for (const [k, stage] of stages.entries()) {
        notify('正在加载第 ' + (k+1) + ' / 4 组模型层…');
        const packed = new Uint8Array(stage.size);
        for (const [source, offset, size] of stage.ranges) {
          readBytes(fh, packed.subarray(offset, offset + size), source);
        }
        const session = await ort.InferenceSession.create(new URL(stage.file,graphDir).href, {
          executionProviders:['wasm'], externalData:[{path:'part_data',data:packed}],
          graphOptimizationLevel:'all',
        });
        try {
          for (let j = begin; j < end; j++) {
            const batch = batches[j];
            notify('正在分析第 ' + (j+1) + ' / ' + batches.length + ' 批文本 · 第 ' + (k+1) + ' / 4 组模型层');
            const {scope} = batch;
            if (k === 0) addEmbeddings(batch, embedding, fh);
            const inputs = Object.fromEntries(stage.inputs.map(key => [key, scope[key]]));
            Object.assign(scope, await session.run(inputs));
            if (k === stages.length - 1) {
              for (const [row, job] of batch.jobs.entries()) {
                const logit = Number(scope.logits.data[row]);
                if (!Number.isFinite(logit)) throw new Error('设备返回了无效的模型分数。');
                const denominator = 1 + Math.exp(-logit);
                if (!batchSentences) rememberChunk(job.key, denominator);
                contributions[job.index][job.chunk] = job.weight / denominator;
                if (--remaining[job.index] === 0) finish(job.index, batchSentences);
              }
            }
            for (const key of Object.keys(scope)) {
              if (!futureInputs[k].has(key)) { scope[key].dispose(); delete scope[key]; }
            }
          }
        } finally { await session.release(); }
      }
      begin = end;
    }
  } finally { fh.close(); }
  return results;
}
