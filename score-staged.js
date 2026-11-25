// Low-memory fp32 inference. The four groups preserve the ONNX layer order.
// A read-only OPFS sync handle provides just the weights needed by each group.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.wasm.min.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
const graphDir = new URL('./stages/', import.meta.url);
const layout = fetch(new URL('layout.json', graphDir)).then(response => {
  if (!response.ok) throw Error('无法加载模型结构。');
  return response.json();
});

function readBytes(handle, target, offset) {
  let n = 0;
  while (n < target.byteLength) {
    const count = handle.read(target.subarray(n), {at: offset + n});
    if (!count) throw new Error('缓存文件不完整。');
    n += count;
  }
}

export async function classifyMany(texts, tokenizer, fileHandle, notify) {
  const {embedding, stages} = await layout;
  const jobs = [];
  const results = texts.map(() => ({score: 0, chunks: 0}));
  // A fresh handle must be closed even on failure. OPFS read-only mode permits
  // simultaneous inference by separate tabs without any write access to weights.
  const fh = await fileHandle.createSyncAccessHandle({mode: 'read-only'});
  try {
    for (const [index, text] of texts.entries()) {
      const ids = tokenizer(text, {add_special_tokens:false, return_tensor:false}).input_ids;
      if (!ids.length) throw new Error('这段文字无法分词。');
      const chunks = Math.ceil(ids.length / 766);
      results[index].chunks = chunks;
      const width = Math.ceil(ids.length / chunks);
      for (let start=0; start<ids.length; start+=width) {
        const part = ids.slice(start,start+width);
        const tokens = [tokenizer.bos_token_id, ...part, tokenizer.sep_token_id];
        jobs.push({index,weight:part.length/ids.length,tokens,scope:{}});
      }
    }
    // Bound intermediate activation memory for long essays. The ONNX graph
    // carries attention indices and masks across each six-layer group.
    let begin=0;
    while (begin<jobs.length) {
      let end=begin, tokensInWindow=0;
      while (end<jobs.length && tokensInWindow+jobs[end].tokens.length<=1536) {
        tokensInWindow+=jobs[end].tokens.length;
        end++;
      }
      for (const [k, stage] of stages.entries()) {
        notify('正在加载第 ' + (k+1) + ' / 4 组模型层…');
        const packed = new Uint8Array(stage.size);
        for (const [source,offset,size] of stage.ranges) {
          readBytes(fh,packed.subarray(offset,offset+size),source);
        }
        const session = await ort.InferenceSession.create(new URL(stage.file,graphDir).href, {
          executionProviders:['wasm'], externalData:[{path:'part_data',data:packed}],
          graphOptimizationLevel:'all',
        });
        try {
          for (let j=begin;j<end;j++) {
            const job=jobs[j];
            notify('正在分析文本块 ' + (j+1) + ' / ' + jobs.length + ' · 第 ' + (k+1) + ' / 4 组模型层');
            const {scope,tokens}=job;
            if (k===0) {
              const data = new Float32Array(tokens.length*1024);
              const bytes = new Uint8Array(data.buffer);
              tokens.forEach((id,n)=>readBytes(fh,bytes.subarray(n*4096,(n+1)*4096),id*4096));
              scope[embedding] = new ort.Tensor('float32',data,[1,tokens.length,1024]);
              scope.attention_mask = new ort.Tensor('int64',new BigInt64Array(tokens.length).fill(1n),[1,tokens.length]);
            }
            const inputs = Object.fromEntries(stage.inputs.map(key=>[key,scope[key]]));
            Object.assign(scope,await session.run(inputs));
            if (k===stages.length-1) {
              const logit = Number(scope.logits.data[0]);
              if (!Number.isFinite(logit)) throw new Error('设备返回了无效的模型分数。');
              results[job.index].score += job.weight/(1+Math.exp(-logit));
            }
            const future = new Set(stages.slice(k+1).flatMap(s=>s.inputs));
            for (const key of Object.keys(scope)) {
              if (!future.has(key)) { scope[key].dispose(); delete scope[key]; }
            }
          }
        } finally { await session.release(); }
      }
      begin=end;
    }
  } finally { fh.close(); }
  return results;
}
