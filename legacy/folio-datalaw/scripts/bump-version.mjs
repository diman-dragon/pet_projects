// Bumps the last digit in package.json (0.0.1 -> 0.0.2) and re-syncs. Minor/major only on owner's command.
import fs from 'node:fs';
import { execSync } from 'node:child_process';
const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
const p = pkg.version.split('.').map(Number);
p[2] += 1;
pkg.version = p.join('.');
fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2) + '\n');
execSync('node scripts/sync-version.mjs', { stdio: 'inherit' });
console.log(`Now add "## ${pkg.version}" to the top of CHANGELOG.md`);
