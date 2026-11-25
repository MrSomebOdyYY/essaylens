import { scoreEssay } from './browser-score.js?v=4';

self.onmessage = async ({data}) => {
  if (data.type !== 'score') return;
  const {id, text, sentences} = data;
  try {
    const result = await scoreEssay(text, message => self.postMessage({type:'progress', id, message}), sentences);
    self.postMessage({type:'result', id, result});
  } catch (error) {
    const message = error instanceof Error ? error.message : '请检查浏览器、网络和可用内存，然后重试。';
    self.postMessage({type:'error', id, message});
  }
};
