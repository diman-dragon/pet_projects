// Contract runner: executes every scripts/checks/*.mjs, prints problems, exits 1 if any.
import fs from 'node:fs';
const dir = new URL('./checks/', import.meta.url);
let bad = 0;
for (const f of fs.readdirSync(dir).filter((n) => n.endsWith('.mjs') && !n.startsWith('_')).sort()) {
  const { default: run } = await import(new URL(f, dir));
  const problems = run();
  console.log(`${problems.length ? 'FAIL' : 'ok  '} ${f.replace('.mjs', '')}`);
  problems.forEach((p) => console.log('     - ' + p));
  bad += problems.length;
}
process.exit(bad ? 1 : 0);
