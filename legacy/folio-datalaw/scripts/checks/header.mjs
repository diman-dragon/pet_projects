// Atomicity: every JS/CSS file declares its single responsibility in the first line.
import fs from 'node:fs';
import { files, GENERATED } from './_walk.mjs';
export default function run() {
  const out = [];
  for (const [dir, exts, re] of [['www/js', ['.js'], /^\/\/ \S/], ['www/css', ['.css'], /^\/\* \S/]]) {
    for (const f of files(dir, exts)) {
      if (GENERATED.includes(f.replaceAll('\\', '/'))) continue;
      if (!re.test(fs.readFileSync(f, 'utf8'))) out.push(`${f}: first line must state the file's responsibility`);
    }
  }
  return out;
}
