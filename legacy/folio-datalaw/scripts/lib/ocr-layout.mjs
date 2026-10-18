// Copies the pieces of the unpacked tesseract npm packages that the app needs into www/vendor/tesseract.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
const walk = (dir) => (fs.existsSync(dir) ? fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)])) : []);
const rank = (f) => (/best_int/.test(f) ? 0 : /fast/.test(f) ? 1 : /4\.0\.0[\\/]/.test(f) ? 2 : 3);
export function copyEngine(pkg, out) {
  fs.mkdirSync(out, { recursive: true });
  const esm = walk(pkg).find((f) => /tesseract\.esm\.min\.js$/.test(f));
  const worker = walk(pkg).find((f) => /dist[\\/]worker\.min\.js$/.test(f));
  if (!esm || !worker) throw new Error('tesseract.js dist files not found');
  fs.copyFileSync(esm, path.join(out, 'tesseract.esm.min.js'));
  fs.copyFileSync(worker, path.join(out, 'worker.min.js'));
}
// Only the LSTM cores are needed (the app runs the LSTM engine); fall back to every core file if the names differ.
export function copyCore(pkg, out) {
  const all = walk(pkg).filter((f) => /\.(wasm\.js|wasm)$/.test(f));
  const lstm = all.filter((f) => /lstm/.test(path.basename(f)));
  const use = lstm.length ? lstm : all;
  if (!use.length) throw new Error('tesseract core files not found');
  fs.mkdirSync(path.join(out, 'core'), { recursive: true });
  use.forEach((f) => fs.copyFileSync(f, path.join(out, 'core', path.basename(f))));
}
// Language data: <code>.traineddata.gz (or raw .traineddata, gzipped here); the smallest good variant wins.
export function copyLang(pkg, code, out) {
  const files = walk(pkg).filter((f) => new RegExp(`${code}\\.traineddata(\\.gz)?$`).test(f)).sort((a, b) => rank(a) - rank(b));
  if (!files.length) throw new Error(`no traineddata for ${code}`);
  fs.mkdirSync(path.join(out, 'lang'), { recursive: true });
  const dst = path.join(out, 'lang', `${code}.traineddata.gz`);
  const buf = fs.readFileSync(files[0]);
  fs.writeFileSync(dst, files[0].endsWith('.gz') ? buf : zlib.gzipSync(buf));
}
