// Version consistency: package.json == version.js == gradle versionName == top CHANGELOG entry.
import fs from 'node:fs';
export default function run() {
  const v = JSON.parse(fs.readFileSync('package.json', 'utf8')).version;
  const out = [];
  const js = fs.readFileSync('www/js/core/version.js', 'utf8');
  if (!js.includes(`'${v}'`)) out.push(`www/js/core/version.js is not ${v} (run npm run version:sync)`);
  const g = fs.readFileSync('android/app/build.gradle', 'utf8');
  if (!g.includes(`versionName "${v}"`)) out.push(`android/app/build.gradle versionName is not ${v}`);
  const top = fs.readFileSync('CHANGELOG.md', 'utf8').match(/^## (\S+)/m)?.[1];
  if (top !== v) out.push(`CHANGELOG.md top entry is ${top}, expected ${v}`);
  return out;
}
