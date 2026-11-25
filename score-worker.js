import { scoreEssay } from './browser-score.js';

self.onmessage = async ({data}) => {
  if (data.type !== 'score') return;
  const {id, text} = data;
  try {
    const result = await scoreEssay(text, message => self.postMessage({type:'progress', id, message}));
    self.postMessage({type:'result', id, result});
  } catch (error) {
    const message = error instanceof Error ? error.message : '请检查浏览器、网络和可用内存，然后重试。';
    self.postMessage({type:'error', id, message});
  }
};
