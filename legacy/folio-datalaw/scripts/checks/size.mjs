// Atomicity: one file = one responsibility => a file must stay small.
import fs from 'node:fs';
import { files, GENERATED } from './_walk.mjs';
const LIMITS = { '.js': 110, '.css': 70, '.mjs': 70 };
export default function run() {
  const out = [];
  for (const [dir, exts] of [['www/js', ['.js']], ['www/css', ['.css']], ['scripts', ['.mjs']]]) {
    for (const f of files(dir, exts)) {
      if (GENERATED.includes(f.replaceAll('\\', '/'))) continue;
      const n = fs.readFileSync(f, 'utf8').split('\n').length;
      const max = LIMITS[exts[0]];
      if (n > max) out.push(`${f}: ${n} lines > ${max} - split by responsibility`);
    }
  }
  return out;
}
