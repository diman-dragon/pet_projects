// Генерирует docs/index.html и docs/privacy.html (RU + EN) из brand/brand.json.
// Включите GitHub Pages: Settings → Pages → Branch: main, папка /docs.
// Полученный адрес (…/privacy.html) вписывается в brand.json → privacyUrl и в Google Play Console.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const b = JSON.parse(fs.readFileSync(path.join(root, 'brand', 'brand.json'), 'utf8'));
const date = new Date().toISOString().slice(0, 10);
fs.mkdirSync(path.join(root, 'docs'), { recursive: true });
fs.copyFileSync(path.join(root, 'www', 'brand', 'dl_logo.png'), path.join(root, 'docs', 'logo.png'));

const css = `body{font:16px/1.6 system-ui,sans-serif;max-width:760px;margin:0 auto;padding:28px 18px;color:#0B0C10;background:#F4F6F9}
h1,h2{line-height:1.25}a{color:${b.colors.accent}}img{height:84px}code{background:#e6e9ef;padding:1px 5px;border-radius:4px}
.lang{display:flex;gap:14px;margin:6px 0 24px}footer{margin-top:40px;font-size:13px;opacity:.7}`;
const page = (title, body) => `<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${title}</title><style>${css}</style></head><body>${body}</body></html>\n`;

fs.writeFileSync(path.join(root, 'docs', 'index.html'), page(b.name, `
<img src="logo.png" alt="${b.shortName}"><h1>${b.name}</h1>
<p>${b.tagline}</p><p><small>${b.taglineEn}</small></p>
<p><a href="privacy.html">Политика конфиденциальности / Privacy Policy</a></p>
<p>Связаться: <a href="mailto:${b.supportEmail}">${b.supportEmail}</a></p>`));

fs.writeFileSync(path.join(root, 'docs', 'privacy.html'), page(`${b.name} — Политика конфиденциальности`, `
<img src="logo.png" alt="${b.shortName}">
<div class="lang"><a href="#ru">Русский</a><a href="#en">English</a></div>

<h1 id="ru">Политика конфиденциальности — ${b.name}</h1>
<p>Дата редакции: ${date}. Правообладатель: ${b.company}. Контакт: <a href="mailto:${b.supportEmail}">${b.supportEmail}</a>.</p>
<h2>Что делает приложение</h2>
<p>${b.name} — камера: фото, видео, ручные настройки, портретный режим и забавные эффекты. Приложение не требует регистрации и не содержит рекламы.</p>
<h2>Доступ к камере и микрофону</h2>
<p>Камера нужна для съёмки; микрофон — только для записи звука в видео. Изображение и звук обрабатываются на устройстве и не отправляются нам или третьим лицам.</p>
<h2>Ваши фото и видео</h2>
<p>Снимки и ролики сохраняются в память телефона (папка <code>DCIM/${b.mediaFolder}</code>). Мы не имеем к ним доступа, не загружаем их на серверы и не используем для обучения моделей.</p>
<h2>Обработка на устройстве</h2>
<p>Распознавание лиц (для эффектов) и выделение человека (для портретного режима) выполняются локально библиотеками Google ML Kit. Результаты распознавания не сохраняются и не передаются.</p>
<h2>Сторонние библиотеки</h2>
<p>Google ML Kit может собирать технические данные для диагностики и аналитики (сведения об устройстве и приложении, показатели производительности) и передавать их в Google по защищённому соединению. Подробнее: <a href="https://developers.google.com/ml-kit/android-data-disclosure">ML Kit — Android data disclosure</a>. Мы не получаем эти данные.</p>
<h2>Хранение настроек</h2>
<p>Настройки приложения (выбранные режимы, фильтры и т. п.) хранятся локально на устройстве и удаляются вместе с приложением.</p>
<h2>Дети</h2><p>Приложение не предназначено для сбора данных о детях и не собирает персональные данные.</p>
<h2>Изменения и контакты</h2><p>При изменении политики мы обновим эту страницу. Вопросы: <a href="mailto:${b.supportEmail}">${b.supportEmail}</a>.</p>

<h1 id="en">Privacy Policy — ${b.name}</h1>
<p>Last updated: ${date}. Publisher: ${b.company}. Contact: <a href="mailto:${b.supportEmail}">${b.supportEmail}</a>.</p>
<h2>What the app does</h2><p>${b.name} is a camera app: photo, video, manual controls, portrait mode and fun effects. No account is required and there are no ads.</p>
<h2>Camera and microphone</h2><p>The camera is used to take photos and video; the microphone is used only to record sound in videos. Images and audio are processed on your device and are not sent to us or to third parties.</p>
<h2>Your photos and videos</h2><p>Media are saved to your device storage (<code>DCIM/${b.mediaFolder}</code>). We cannot access them, never upload them and never use them to train models.</p>
<h2>On-device processing</h2><p>Face detection (effects) and person segmentation (portrait mode) run locally using Google ML Kit. Results are not stored or transmitted.</p>
<h2>Third-party libraries</h2><p>Google ML Kit may collect technical diagnostics and usage analytics (device and app information, performance metrics) and send them to Google over an encrypted connection. See <a href="https://developers.google.com/ml-kit/android-data-disclosure">ML Kit — Android data disclosure</a>. We do not receive this data.</p>
<h2>Settings</h2><p>App settings are stored locally and removed when the app is uninstalled.</p>
<h2>Children</h2><p>The app is not directed at children and does not collect personal data.</p>
<h2>Changes and contact</h2><p>We will update this page if the policy changes. Questions: <a href="mailto:${b.supportEmail}">${b.supportEmail}</a>.</p>
<footer>© ${new Date().getFullYear()} ${b.company}</footer>`));
console.log('docs/index.html, docs/privacy.html — готово');
