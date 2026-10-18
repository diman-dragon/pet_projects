// Shared helper (not a check, runner skips "_" files): lists project source files.
import fs from 'node:fs';
import path from 'node:path';
export function files(dir, exts) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) return e.name === 'vendor' ? [] : files(p, exts);
    return exts.includes(path.extname(e.name)) ? [p] : [];
  });
}
export const GENERATED = ['www/js/core/version.js'];
