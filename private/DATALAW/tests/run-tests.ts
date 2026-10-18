import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { APP_ROLES, canAssignRole, canManageTargetUser } from '../src/services/auth/permissions.ts';
import { isPathInside, normalizePath } from '../src/services/fs/pathAccessCore.ts';
import { CaseNamingService } from '../src/services/naming/CaseNamingService.ts';
import { AppealNamingService } from '../src/services/naming/AppealNamingService.ts';

function run(name: string, fn: () => void) {
  try {
    fn();
    console.log(`PASS ${name}`);
  } catch (error) {
    console.error(`FAIL ${name}`);
    throw error;
  }
}

run('superadmin cannot manage own record', () => {
  assert.equal(
    canManageTargetUser(
      { id: 1, role: APP_ROLES.SUPERADMIN },
      { id: 1, role: APP_ROLES.SUPERADMIN },
    ),
    false,
  );
});

run('admin cannot manage another admin', () => {
  assert.equal(
    canManageTargetUser({ id: 1, role: APP_ROLES.ADMIN }, { id: 2, role: APP_ROLES.ADMIN }),
    false,
  );
});

run('staff UI cannot assign superadmin role', () => {
  assert.equal(canAssignRole({ role: APP_ROLES.SUPERADMIN }, APP_ROLES.SUPERADMIN), false);
  assert.equal(canAssignRole({ role: APP_ROLES.ADMIN }, APP_ROLES.SUPERADMIN), false);
});

run('normalizePath lowers case and trims trailing slash', () => {
  assert.equal(normalizePath('D:/Dorlane/Clients/'), 'd:\\Dorlane\\clients');
});

run('isPathInside returns true for nested workspace path', () => {
  assert.equal(isPathInside('D:\\Dorlane', 'D:\\Dorlane\\Clients\\ACME'), true);
});

run('isPathInside returns false for foreign path', () => {
  assert.equal(isPathInside('D:\\Dorlane', 'D:\\Other\\file.txt'), false);
});

run('case folder names ignore status and stay stable', () => {
  const generated = CaseNamingService.contract.generateFolderName({ title: 'Тестовое дело', status: 'CLOSED' });
  assert.equal(generated, 'TESTOVOE_DELO');
  assert.equal(CaseNamingService.litigationContract.generateFolderName({ title: 'Тестовое дело', status: 'CLOSED' }), 'TESTOVOE_DELO');
});

run('appeal folder names do not include status', () => {
  const generated = AppealNamingService.contract.generateFolderName({
    id: 12,
    title: 'Обращение по делу',
    status: 'CLOSED',
    appeal_date: '2026-07-04',
  });
  assert.match(generated, /01_DOC_/);
  assert.equal(generated.includes('_CLOSED'), false);
});

run('connection manager uses the unified Rust-backed DB path', () => {
  const connectionManagerSource = readFileSync(new URL('../src/services/db/ConnectionManager.ts', import.meta.url), 'utf8');
  const packageSource = readFileSync(new URL('../package.json', import.meta.url), 'utf8');
  const tauriCargoSource = readFileSync(new URL('../src-tauri/Cargo.toml', import.meta.url), 'utf8');
  assert.equal(connectionManagerSource.includes('Database.load'), false);
  assert.equal(connectionManagerSource.includes('invoke('), true);
  assert.equal(packageSource.includes('@tauri-apps/plugin-sql'), false);
  assert.equal(tauriCargoSource.includes('tauri-plugin-sql'), false);
});

run('master key service no longer stores password in localStorage', () => {
  const masterKeySource = readFileSync(new URL('../src/services/security/MasterKeyService.ts', import.meta.url), 'utf8');
  assert.equal(masterKeySource.includes('localStorage'), false);
  assert.equal(masterKeySource.includes('bootstrap'), false);
});

console.log('All tests passed');
