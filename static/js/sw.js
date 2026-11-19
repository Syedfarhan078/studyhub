/* =========================================================================
   StudyHub Service Worker — Offline PWA & Academic Asset Caching
   ========================================================================= */

const CACHE_VERSION = 'studyhub-v1';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-pages`;
const NOTES_CACHE = `${CACHE_VERSION}-notes`;

const CORE_STATIC_ASSETS = [
    '/',
    '/offline/',
    '/static/css/custom.css',
    '/static/logo/studyhublogo.png',
    '/static/manifest.json',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js',
    'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css'
];

// 1. Install Event — Pre-cache App Shell
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(STATIC_CACHE).then((cache) => {
            return cache.addAll(CORE_STATIC_ASSETS).catch((err) => {
                console.warn('[SW] Non-critical asset failed to pre-cache:', err);
            });
        }).then(() => self.skipWaiting())
    );
});

// 2. Activate Event — Clean up stale versions
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.filter((key) => key.startsWith('studyhub-') && !key.startsWith(CACHE_VERSION))
                    .map((key) => caches.delete(key))
            );
        }).then(() => self.clients.claim())
    );
});

// 3. Fetch Event — Routing & Caching Strategies
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // Only process GET requests (ignore POST, PUT, DELETE)
    if (request.method !== 'GET') {
        return;
    }

    // A. Note Viewing & PDF Streams (/subjects/notes/<id>/)
    if (url.pathname.includes('/subjects/notes/')) {
        event.respondWith(
            fetch(request)
                .then((networkRes) => {
                    if (networkRes && networkRes.status === 200) {
                        const resClone = networkRes.clone();
                        caches.open(NOTES_CACHE).then((cache) => cache.put(request, resClone));
                    }
                    return networkRes;
                })
                .catch(() => {
                    return caches.match(request).then((cachedRes) => {
                        if (cachedRes) {
                            return cachedRes;
                        }
                        return caches.match('/offline/');
                    });
                })
        );
        return;
    }

    // B. Navigation Requests (HTML Pages: Homepage, Subjects)
    if (request.mode === 'navigate') {
        event.respondWith(
            fetch(request)
                .then((networkRes) => {
                    if (networkRes && networkRes.status === 200) {
                        const resClone = networkRes.clone();
                        caches.open(DYNAMIC_CACHE).then((cache) => cache.put(request, resClone));
                    }
                    return networkRes;
                })
                .catch(() => {
                    return caches.match(request).then((cachedRes) => {
                        if (cachedRes) {
                            return cachedRes;
                        }
                        return caches.match('/offline/');
                    });
                })
        );
        return;
    }

    // C. Static Assets (CSS, JS, Fonts, Images, Icons)
    if (
        url.pathname.startsWith('/static/') ||
        url.hostname.includes('cdn.jsdelivr.net') ||
        url.hostname.includes('fonts.googleapis.com') ||
        url.hostname.includes('fonts.gstatic.com')
    ) {
        event.respondWith(
            caches.match(request).then((cachedRes) => {
                if (cachedRes) {
                    return cachedRes;
                }
                return fetch(request).then((networkRes) => {
                    if (networkRes && networkRes.status === 200) {
                        const resClone = networkRes.clone();
                        caches.open(STATIC_CACHE).then((cache) => cache.put(request, resClone));
                    }
                    return networkRes;
                });
            })
        );
        return;
    }

    // Default: Network with Cache Fallback
    event.respondWith(
        fetch(request).catch(() => caches.match(request))
    );
});

// 4. Background Sync / Pre-caching of "My Desk" Pinned Subjects
self.addEventListener('message', (event) => {
    if (event.data && event.data.action === 'cache_pinned') {
        const urls = event.data.urls || [];
        caches.open(DYNAMIC_CACHE).then((cache) => {
            urls.forEach((url) => {
                fetch(url).then((res) => {
                    if (res.status === 200) {
                        cache.put(url, res);
                    }
                }).catch(() => {});
            });
        });
    }
});
