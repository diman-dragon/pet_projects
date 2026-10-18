import fs from 'node:fs';
import path from 'node:path';

const baseDir = path.resolve('src/i18n/locales');
const baseLanguage = 'ru';
const namespaces = ['common', 'forms', 'documents', 'appeals', 'rag', 'errors'];

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

const base = Object.fromEntries(
  namespaces.map((namespace) => [
    namespace,
    readJson(path.join(baseDir, baseLanguage, `${namespace}.json`)),
  ]),
);

const languages = fs
  .readdirSync(baseDir)
  .filter((entry) => fs.statSync(path.join(baseDir, entry)).isDirectory());
const failures = [];

for (const language of languages) {
  for (const namespace of namespaces) {
    const targetPath = path.join(baseDir, language, `${namespace}.json`);
    if (!fs.existsSync(targetPath)) {
      failures.push(`${language}/${namespace}.json is missing`);
      continue;
    }

    const target = readJson(targetPath);
    const baseKeys = Object.keys(base[namespace]);
    const targetKeys = new Set(Object.keys(target));

    for (const key of baseKeys) {
      if (!targetKeys.has(key)) {
        failures.push(`${language}/${namespace}.json missing key: ${key}`);
      }
    }
  }
}

if (failures.length > 0) {
  console.error('Translation validation failed:\n' + failures.join('\n'));
  process.exit(1);
}

console.log(`Translation validation passed for ${languages.length} languages.`);
