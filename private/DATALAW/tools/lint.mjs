import fs from 'node:fs';
import path from 'node:path';

const roots = ['src', 'tests', 'tools'];
const exts = new Set(['.ts', '.tsx', '.js', '.mjs']);
const failures = [];

function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'node_modules' || entry.name === 'dist' || entry.name === 'docs') continue;
      walk(fullPath);
      continue;
    }

    if (!exts.has(path.extname(entry.name))) continue;

    const content = fs.readFileSync(fullPath, 'utf8');
    const lines = content.split(/\r?\n/);
    lines.forEach((line, index) => {
      if (/\s+$/.test(line)) {
        failures.push(`${fullPath}:${index + 1} trailing whitespace`);
      }
    });
  }
}

for (const root of roots) {
  walk(path.resolve(root));
}

if (failures.length > 0) {
  console.error('Lint failed:\n' + failures.join('\n'));
  process.exit(1);
}

console.log('Lint passed.');
