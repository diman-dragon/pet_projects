// src/services/search/searchSchemaBootstrap.ts
//
// Вызывается один раз после завершения DatabaseSchemaStep/DatabaseMigrationStep.
// Выполняет:
//   1. PRAGMA optimize для всех трёх БД (после массовой индексации или миграций)
//   2. Проверяет, что системные category_rules засеяны
//   3. ИТЕРАЦИЯ 4: warm-up — загружает кеш подсказок (recent activity)
//      чтобы первый открытый поиск был мгновенным

import { ConnectionManager } from '../db/ConnectionManager';
import { createLogger } from '../logger/Logger';
import { SearchSuggestionsService } from './SearchSuggestionsService';

const log = createLogger('searchSchemaBootstrap');

const SYSTEM_RULES_SEED = `
  INSERT OR IGNORE INTO category_rules (rule_type, pattern, category, priority, is_system) VALUES
    ('extension', 'pdf',   'Documents', 10, 1),
    ('extension', 'docx',  'Documents', 10, 1),
    ('extension', 'doc',   'Documents', 10, 1),
    ('extension', 'txt',   'Documents',  5, 1),
    ('extension', 'md',    'Notes',      5, 1),
    ('extension', 'tsv',   'Data',       5, 1),
    ('extension', 'csv',   'Data',       5, 1),
    ('path_pattern', '%/contracts/%',    'Contracts',   20, 1),
    ('path_pattern', '%/договор%',       'Contracts',   20, 1),
    ('path_pattern', '%/суд%',           'Litigation',  20, 1),
    ('path_pattern', '%/court%',         'Litigation',  20, 1),
    ('path_pattern', '%/templates/%',    'Templates',   15, 1),
    ('path_pattern', '%/шаблон%',        'Templates',   15, 1),
    ('path_pattern', '%/appeals/%',      'Appeals',     15, 1),
    ('path_pattern', '%/обращен%',       'Appeals',     15, 1),
    ('default',      '*',               'General',      0, 1);
`;

/**
 * Запускает PRAGMA analysis_limit + optimize на соединении.
 */
async function runOptimize(key: string): Promise<void> {
  if (!ConnectionManager.exists(key)) return;
  try {
    const db = ConnectionManager.get(key);
    await db.execute('PRAGMA analysis_limit=1000;');
    await db.execute('PRAGMA optimize;');
    log.debug(`PRAGMA optimize done [${key}]`);
  } catch (err) {
    log.warn(`PRAGMA optimize failed [${key}]`, err);
  }
}

/**
 * Проверяет и при необходимости засеивает системные category_rules.
 */
async function ensureCategoryRules(): Promise<void> {
  if (!ConnectionManager.exists('kb_main')) return;
  try {
    const db = ConnectionManager.get('kb_main');
    const rows = await db.select<Array<{ cnt: number }>>(
      `SELECT COUNT(*) AS cnt FROM category_rules WHERE is_system = 1`,
    );
    const count = rows?.[0]?.cnt ?? 0;
    if (count === 0) {
      await db.execute(SYSTEM_RULES_SEED);
      log.info('Seeded system category_rules');
    }
  } catch (err) {
    log.warn('ensureCategoryRules failed', err);
  }
}

/**
 * ИТЕРАЦИЯ 4: Warm-up кеша подсказок.
 * Загружаем recent activity заранее — при первом открытии поиска
 * подсказки будут доступны мгновенно.
 */
async function warmUpSuggestionsCache(): Promise<void> {
  try {
    await SearchSuggestionsService.getRecentActivity();
    log.debug('Suggestions cache warmed up');
  } catch {
    // Некритично — suggestions появятся при первом открытии
  }
}

/**
 * Основная точка входа. Вызывать из AppInitializer после завершения
 * DatabaseMigrationStep и DatabaseSchemaStep.
 */
export async function runSearchSchemaBootstrap(): Promise<void> {
  await ensureCategoryRules();
  await Promise.allSettled([
    runOptimize('kb_main'),
    runOptimize('clients_main'),
    runOptimize('user_system'),
  ]);
  // Warm-up фоново — не блокируем загрузку приложения
  warmUpSuggestionsCache().catch(() => {});
  log.info('Search schema bootstrap complete');
}
