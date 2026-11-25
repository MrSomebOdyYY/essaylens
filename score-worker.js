import { scoreWhole, scoreEssay } from './browser-score.js?v=5';

self.onmessage = async ({data}) => {
  if (data.type !== 'score') return;
  const {id, text, sentences} = data;
  const send = (type, fields) => self.postMessage({type, id, ...fields});
  const progress = message => send('progress', {message});
  try {
    const whole = await scoreWhole(text, progress);
    send('whole', {result:whole});
    if (sentences) {
      await scoreEssay(text, progress, sentences, (index, row) => {
        send('sentence', {index, row});
      });
    }
    send('done', {});
  } catch (error) {
    const message = error instanceof Error ? error.message : '请检查浏览器、网络和可用内存，然后重试。';
    send('error', {message});
  }
};
