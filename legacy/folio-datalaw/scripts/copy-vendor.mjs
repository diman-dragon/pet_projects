// node_modules/pdfjs-dist (legacy build) -> www/vendor/pdfjs. Generated, git-ignored.
import fs from 'node:fs';
const src = 'node_modules/pdfjs-dist';
const dst = 'www/vendor/pdfjs';
if (!fs.existsSync(src)) { console.log('pdfjs-dist not installed, skip'); process.exit(0); }
fs.rmSync(dst, { recursive: true, force: true });
fs.mkdirSync(dst, { recursive: true });
const copy = (from, to) => fs.cpSync(`${src}/${from}`, `${dst}/${to}`, { recursive: true });
copy('legacy/build/pdf.min.mjs', 'pdf.min.mjs');
copy('legacy/build/pdf.worker.min.mjs', 'pdf.worker.min.mjs');
copy('cmaps', 'cmaps');
copy('standard_fonts', 'standard_fonts');
copy('wasm', 'wasm');
copy('iccs', 'iccs');
fs.copyFileSync(`${src}/LICENSE`, `${dst}/LICENSE`);
console.log('pdfjs vendored to', dst);
