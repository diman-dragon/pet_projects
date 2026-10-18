// Создаёт РЕАЛЬНЫЙ ключ подписи DataLaw Camera и готовит секреты для GitHub.
//   npm run keystore            — создать ключ (нужен JDK: keytool)
//   npm run keystore -- --gh    — дополнительно записать секреты в репозиторий через GitHub CLI (gh)
// ВАЖНО: ключ определяет, будет ли приложение ОБНОВЛЯТЬСЯ. Потеряете — обновить уже
// установленные копии (вне Google Play) станет невозможно. Сделайте резервную копию файла и пароля.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'signing', 'datalaw-release.p12');
const alias = 'datalaw';
const useGh = process.argv.includes('--gh');
const force = process.argv.includes('--force');

if (fs.existsSync(out) && !force) {
  console.error(`Уже есть ${path.relative(root, out)}. Не перезаписываю (--force чтобы пересоздать — обновления старых установок сломаются!).`);
  process.exit(1);
}
const pass = crypto.randomBytes(18).toString('base64url');
const r = spawnSync('keytool', [
  '-genkeypair', '-keystore', out, '-storetype', 'PKCS12', '-alias', alias,
  '-keyalg', 'RSA', '-keysize', '4096', '-validity', '10000',
  '-storepass', pass, '-keypass', pass, '-dname', 'CN=DataLaw Camera, O=DataLaw',
], { stdio: 'inherit' });
if (r.error || r.status !== 0) {
  console.error('keytool не найден или завершился с ошибкой. Установите JDK 21 (Temurin) и повторите.');
  process.exit(1);
}
const b64 = fs.readFileSync(out).toString('base64');
fs.writeFileSync(out + '.b64', b64);
const fp = spawnSync('keytool', ['-list', '-keystore', out, '-storepass', pass], { encoding: 'utf8' }).stdout || '';

console.log(`
Ключ создан: ${path.relative(root, out)}   (в git не попадёт — см. .gitignore)

Секреты GitHub (Settings → Secrets and variables → Actions → New repository secret):
  DL_KEYSTORE_BASE64    = содержимое файла ${path.relative(root, out + '.b64')}
  DL_KEYSTORE_PASSWORD  = ${pass}
  DL_KEY_ALIAS          = ${alias}
  (DL_KEY_PASSWORD не нужен — совпадает с паролем хранилища)

${fp.split('\n').filter((l) => /fingerprint|Certificate/i.test(l)).join('\n')}

СОХРАНИТЕ файл ${path.basename(out)} и пароль в менеджере паролей — это единственная копия.
`);
if (useGh) {
  const set = (name, val) => spawnSync('gh', ['secret', 'set', name, '--body', val], { stdio: 'inherit' });
  set('DL_KEYSTORE_BASE64', b64); set('DL_KEYSTORE_PASSWORD', pass); set('DL_KEY_ALIAS', alias);
  console.log('Секреты записаны через gh.');
}
