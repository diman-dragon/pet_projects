import { execSync } from 'node:child_process';

function safe(command) {
  try {
    return execSync(command, { stdio: ['ignore', 'pipe', 'ignore'] })
      .toString('utf8')
      .trim();
  } catch {
    return '';
  }
}

const latestTag = safe('git describe --tags --abbrev=0');
const range = latestTag ? `${latestTag}..HEAD` : 'HEAD';
const commits = safe(`git log ${range} --pretty=format:"- %s"`).split('\n').filter(Boolean);

const notes = [
  '# Release Notes',
  '',
  latestTag ? `Changes since ${latestTag}:` : 'Changes in current revision:',
  '',
  ...(commits.length > 0 ? commits : ['- No commits found']),
].join('\n');

process.stdout.write(`${notes}\n`);
