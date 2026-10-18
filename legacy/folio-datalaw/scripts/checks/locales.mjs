// ru.js and en.js must define the same keys.
import fs from 'node:fs';
const keys = (f) => new Set([...fs.readFileSync(f, 'utf8').matchAll(/^\s*'([^']+)'\s*:/gm)].map((m) => m[1]));
export default function run() {
  const ru = keys('www/js/core/locales/ru.js');
  const en = keys('www/js/core/locales/en.js');
  return [
    ...[...ru].filter((k) => !en.has(k)).map((k) => `en.js misses "${k}"`),
    ...[...en].filter((k) => !ru.has(k)).map((k) => `ru.js misses "${k}"`),
  ];
}
