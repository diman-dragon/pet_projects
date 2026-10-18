// Every relative static import in www/js must resolve to an existing file.
import fs from 'node:fs';
import path from 'node:path';
import { files } from './_walk.mjs';
export default function run() {
  const out = [];
  for (const f of files('www/js', ['.js'])) {
    const src = fs.readFileSync(f, 'utf8');
    for (const m of src.matchAll(/(?:from|import)\s+['"](\.[^'"]+)['"]/g)) {
      if (!fs.existsSync(path.resolve(path.dirname(f), m[1]))) out.push(`${f}: cannot resolve ${m[1]}`);
    }
  }
  return out;
}
