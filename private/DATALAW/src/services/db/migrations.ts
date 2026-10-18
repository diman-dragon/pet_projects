// src/services/db/migrations.ts
//
// Как добавить миграцию:
//   1. Добавь объект в конец нужного массива в `src/services/db/migrations/*`
//   2. version должен быть строго больше предыдущего для того же key
//   3. key — имя соединения: 'clients_main' | 'user_system' | 'kb_main'
//   4. sql — один SQL-statement; для нескольких операций добавь несколько объектов
//   5. Никогда не редактируй существующие миграции — только добавляй новые
//
// Все публичные импорты остаются совместимыми через этот файл-агрегатор.

export { MIGRATIONS } from './migrations/index';
export type { Migration } from './migrations/index';
