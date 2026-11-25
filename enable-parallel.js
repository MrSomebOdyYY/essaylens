// An isolated top-level page lets the model use several CPU threads. On static
// hosts our service worker adds the headers on the next navigation.
async function prepareParallelScoring() {
  if (!isSecureContext || window.top !== window || !('serviceWorker' in navigator)) return;
  const workers = navigator.serviceWorker;
  if (crossOriginIsolated || (navigator.hardwareConcurrency || 1) <= 1) return;

  const field = document.getElementById('essay');
  let interacted = false;
  const markInteraction = () => { interacted = true; };
  field.addEventListener('input', markInteraction, {once:true});
  document.getElementById('run').addEventListener('click', markInteraction, {once:true});
  // Leave restored drafts and in-flight work in place. If reloading is not
  // possible, this visit can still use the single-threaded model.
  await new Promise(resolve => {
    if (document.readyState === 'complete') resolve();
    else window.addEventListener('pageshow', resolve, {once:true});
  });
  await workers.register('./parallel-worker.js', {scope:'./', updateViaCache:'none'});
  await workers.ready;
  const scriptURL = new URL('./parallel-worker.js', import.meta.url).href;
  const reloadWhenIdle = () => {
    if (workers.controller?.scriptURL !== scriptURL || interacted || field.value ||
        document.getElementById('run').disabled || performance.getEntriesByType('navigation')[0]?.type === 'back_forward') return;
    const key = 'essaylens-parallel-navigation';
    if (sessionStorage.getItem(key)) return;
    sessionStorage.setItem(key, '1');
    location.reload();
  };
  if (workers.controller?.scriptURL === scriptURL) reloadWhenIdle();
  else workers.addEventListener('controllerchange', () => {
    try { reloadWhenIdle(); } catch { /* Storage restrictions can disable the one-time refresh. */ }
  }, {once:true});
}

prepareParallelScoring().catch(() => {}); // Optional acceleration.
