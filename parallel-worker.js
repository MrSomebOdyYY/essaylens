// Only add isolation headers to this site's documents and worker entry points.
// Model downloads and essays never pass through this handler.
self.addEventListener('install', event => event.waitUntil(self.skipWaiting()));
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {
  const {request} = event;
  if (request.method !== 'GET' || !request.url.startsWith(self.registration.scope)) return;
  if (request.mode !== 'navigate' && request.destination !== 'worker' && request.destination !== 'sharedworker') return;
  event.respondWith((async () => {
    const source = await fetch(request);
    if (!source.ok || source.redirected) return source;
    const headers = new Headers(source.headers);
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    return new Response(source.body, {status:source.status, statusText:source.statusText, headers});
  })());
});
