// Проверка согласованности бренда/конфига. В CI запускается перед сборкой.
//   node scripts/check-brand.mjs            — жёсткие ошибки + предупреждения о заглушках
//   node scripts/check-brand.mjs --strict   — заглушки (example.com, CHANGE_ME) тоже считаются ошибкой (для релиза в Google Play)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const j = (p) => JSON.parse(fs.readFileSync(path.join(root, p), 'utf8'));
const cfg = j('capacitor.config.json'), brand = j('brand/brand.json'), pkg = j('package.json');
const strict = process.argv.includes('--strict');
const errors = [], warns = [];

if (cfg.appId !== brand.appId) errors.push(`appId: capacitor.config.json=${cfg.appId} / brand.json=${brand.appId}`);
if (cfg.appName !== brand.name) errors.push(`appName: capacitor.config.json="${cfg.appName}" / brand.json="${brand.name}"`);
if (!/^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$/.test(brand.appId)) errors.push(`некорректный appId: ${brand.appId}`);
if (!/^\d+\.\d+\.\d+$/.test(pkg.version)) errors.push(`version в package.json должен быть X.Y.Z, сейчас ${pkg.version}`);
for (const f of ['android-res/drawable-nodpi/dl_ic_foreground.png', 'android-res/drawable-nodpi/dl_ic_background.png',
  'android-res/drawable-nodpi/dl_ic_monochrome.png', 'android-res/drawable-nodpi/dl_splash_icon.png',
  'www/brand/dl_logo.png', 'www/brand/dl_mark_white.png', 'www/brand/dl_watermark.png',
  'signing/datalaw-dev.p12']) if (!fs.existsSync(path.join(root, f))) errors.push(`нет файла ${f}`);

const placeholder = (v) => /example\.com|CHANGE_ME/i.test(v || '');
for (const k of ['website', 'privacyUrl', 'supportEmail']) if (placeholder(brand[k])) warns.push(`brand/brand.json → ${k} всё ещё заглушка: ${brand[k]}`);

const ci = !!process.env.GITHUB_ACTIONS;
for (const w of warns) console.log(ci ? `::${strict ? 'error' : 'warning'}::${w}` : `⚠  ${w}`);
for (const e of errors) console.log(ci ? `::error::${e}` : `✖  ${e}`);
if (errors.length || (strict && warns.length)) process.exit(1);
console.log('Бренд/конфиг: OK' + (warns.length ? ' (есть предупреждения)' : ''));
