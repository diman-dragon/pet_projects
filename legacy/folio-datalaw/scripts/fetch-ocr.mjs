// Downloads the offline OCR engine (tesseract.js + core + rus/eng/ita/srp/srp_latn data) from npm into www/vendor/tesseract. Best effort.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { copyEngine, copyCore, copyLang } from './lib/ocr-layout.mjs';
const out = 'www/vendor/tesseract';
const LANGS = ['rus', 'eng', 'ita', 'srp', 'srp_latn'];
if (fs.existsSync(`${out}/ready.json`) && !process.argv.includes('--force')) { console.log('OCR already vendored (use --force to refetch)'); process.exit(0); }
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'folio-ocr-'));
function unpack(spec) {
  const name = execFileSync(npm, ['pack', spec, '--pack-destination', tmp, '--silent'], { encoding: 'utf8', shell: process.platform === 'win32' }).trim().split('\n').pop();
  const dir = path.join(tmp, `u${fs.readdirSync(tmp).length}`);
  fs.mkdirSync(dir);
  execFileSync('tar', ['-xzf', path.join(tmp, name), '-C', dir]);
  return path.join(dir, 'package');
}
try {
  fs.rmSync(out, { recursive: true, force: true });
  fs.mkdirSync(out, { recursive: true });
  const engine = unpack('tesseract.js@5');
  copyEngine(engine, out);
  if (fs.existsSync(`${engine}/LICENSE`)) fs.copyFileSync(`${engine}/LICENSE`, `${out}/LICENSE`);
  copyCore(unpack('tesseract.js-core@5'), out);
  const langs = [];
  for (const code of LANGS) {
    try { copyLang(unpack(`@tesseract.js-data/${code}`), code, out); langs.push(code); } catch (err) { console.warn(`OCR language ${code} skipped: ${err.message}`); }
  }
  if (!langs.length) throw new Error('no language data');
  fs.writeFileSync(`${out}/ready.json`, JSON.stringify({ engine: 5, langs, core: fs.readdirSync(`${out}/core`) }));
  console.log('OCR vendored:', langs.join(', '));
} catch (err) {
  fs.rmSync(out, { recursive: true, force: true });
  console.warn(`OCR engine not installed (${err.message}). The app works without it; run "npm run fetch-ocr" with internet access.`);
} finally { fs.rmSync(tmp, { recursive: true, force: true }); }
