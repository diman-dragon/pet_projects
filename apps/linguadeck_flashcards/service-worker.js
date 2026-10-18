// service-worker.js
// Одна задача: закэшировать "оболочку" приложения при первой загрузке,
// чтобы дальше всё работало офлайн.

const CACHE_NAME = "flashcards-v12";
const APP_SHELL = [
  "./",
  "./index.html",
  "./manifest.json",
  "./css/tokens.css",
  "./css/layout.css",
  "./css/components.css",
  "./js/util.js",
  "./js/schema.js",
  "./js/seed.js",
  "./js/db.js",
  "./js/mediaStore.js",
  "./js/cardManager.js",
  "./js/srs.js",
  "./js/study.js",
  "./js/stats.js",
  "./js/deckManager.js",
  "./js/editCard.js",
  "./js/navigation.js",
  "./js/flip.js",
  "./js/render.js",
  "./js/batchAdd.js",
  "./js/addCard.js",
  "./js/addDialog.js",
  "./js/editDialog.js",
  "./js/deckDialog.js",
  "./js/directionDialog.js",
  "./js/feedback.js",
  "./js/backup.js",
  "./js/importDialog.js",
  "./js/menu.js",
  "./js/actions.js",
  "./js/googleDriveSync.js",
  "./js/syncDialog.js",
  "./js/app.js",
  "./vendor/sql.js/sql-wasm.js",
  "./vendor/sql.js/sql-wasm.wasm",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./media/english-b1/ubiquitous.jpg",
  "./media/english-b1/ubiquitous.mp3",
  "./media/capitals/australia.jpg",
  "./media/capitals/canada.mp3"
];

self.addEventListener("install", function (event) {
  event.waitUntil(
    caches.open(CACHE_NAME).then(function (cache) {
      return cache.addAll(APP_SHELL);
    })
  );
  self.skipWaiting();
});

self.addEventListener("activate", function (event) {
  event.waitUntil(
    caches.keys().then(function (names) {
      return Promise.all(
        names
          .filter(function (name) { return name !== CACHE_NAME; })
          .map(function (name) { return caches.delete(name); })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener("fetch", function (event) {
  event.respondWith(
    caches.match(event.request).then(function (cached) {
      return cached || fetch(event.request);
    })
  );
});
