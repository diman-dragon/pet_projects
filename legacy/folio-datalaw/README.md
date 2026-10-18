# Folio (folio-datalaw)

Универсальный просмотрщик документов для Android на Capacitor. Сейчас — только PDF, остальные форматы будут отдельными модулями.
Правила разработки — в [CONTRACT.md](CONTRACT.md), требования магазина — в [STORE.md](STORE.md).

## Быстрый старт (VS Code)
```
npm install          # ставит зависимости и копирует pdf.js в www/vendor
npm run dev          # http://localhost:5173 — просмотр в браузере (F5 в VS Code = Chrome)
npm run check        # проверка контракта (делать после КАЖДОГО изменения)
npm run bump         # версия +1 в последней цифре, затем запись в CHANGELOG.md
```
Задачи VS Code: `Terminal → Run Task` (dev server, check contract, bump version, sync android, build debug apk).

## Сборка
- **GitHub Actions** (`.github/workflows/build.yml`): push в `main` → артефакт `folio-datalaw_<версия>_debug.apk`.
  Если заданы секреты — подписанные `folio-datalaw_<версия>.apk` и `.aab`.
- Debug подписывается фиксированным ключом `android/debug.keystore`, поэтому новая сборка ставится поверх старой.
- **Секреты репозитория** для release: `KEYSTORE_BASE64`, `KEYSTORE_PASSWORD`, `KEY_ALIAS`, `KEY_PASSWORD`.
  Создать ключ: `keytool -genkeypair -v -keystore upload.jks -alias upload -keyalg RSA -keysize 2048 -validity 10000`,
  в base64: `base64 -w0 upload.jks`. **Храните ключ в надёжном месте.**
- Локально: `npm run sync`, затем `cd android && ./gradlew assembleDebug` (нужны JDK 21 и Android SDK).

## Структура `www/js` (1 файл = 1 ответственность)
```
main.js                 точка входа
core/                   storage, platform, bus, theme, i18n(+locales), night, format, version(генерируется)
data/                   db.js (IndexedDB), library.js (записи + файлы)
io/                     pick-file.js, launch-intent.js («Открыть с помощью Folio»), folder.js (плагин Folder)
app/                    open-file, doc-source, folder-sync/-watch (папка -> библиотека), cover-loader (обложки)
ui/                     tabs, back, toast, icon, home-screen, settings-screen, viewer-screen/-toolbar/-night
formats/registry.js     реестр модулей;  formats/index.js — единственное место подключения
formats/pdf/            pdfjs, open-doc, render-page, page-list, page-tracker, zoom, mount, index
```

## Как добавить формат
1. Папка `www/js/formats/<fmt>/` с модулем `{ id, extensions, mimes, mount(ctx) }`.
   `mount({container, scroller, blob, startPage, onPage})` возвращает `{ zoomBy(f), goTo(n), destroy() }`.
2. В `formats/index.js`: один `import` и один `register(...)`.
3. Для Android — добавить MIME в `intent-filter` в `AndroidManifest.xml`.

## Темы
`www/css/theme.css`: светлая — песочный фон + тёмно-коричневые панели; тёмная — чистый чёрный (AMOLED). Все цвета — токены в одном файле.

## Известные ограничения
- Имя файла при «Открыть с помощью» недоступно из WebView (content://) — подставляется «Document <дата>.pdf». Решение: маленький нативный плагин, позже.
- Файл целиком читается в память; очень большие PDF (сотни МБ) могут не открыться.
- До первого рендера высота страниц берётся по первой странице (возможен небольшой «скачок» в документах с разными форматами страниц).
- Запрос пароля PDF — через системный `prompt()`.

## Папка библиотеки
Нативный плагин `android/.../FolderPlugin.java` (Storage Access Framework): `pick`, `list`, `stage`. Права на папку сохраняются между запусками. В браузере (`npm run dev`) выбор папки недоступен.

## Распознавание текста (OCR) и картинки в PDF

- OCR работает офлайн: `npm install` (скрипт `postinstall`) скачивает tesseract.js 5, ядро и данные языков (rus, eng, ita, srp, srp_latn) в `www/vendor/tesseract` (папка в `.gitignore`). Для повторной загрузки: `npm run fetch-ocr`. Нужен интернет и утилита `tar`; без них приложение собирается и работает, но кнопка распознавания сообщает, что модуль не установлен. Размер APK растёт примерно на 40-60 МБ.
- Кнопка «Картинки в PDF» (библиотека) собирает PDF без внешних библиотек (`data/pdf-writer.js`).

## Форматы

PDF и FB2 (`.fb2`, `.fb2.zip`, `.fbz`). Новый формат подключается модулем в `www/js/formats/<имя>/` и одной строкой в `formats/index.js`; библиотека, папка и «Открыть с помощью» используют реестр форматов.
