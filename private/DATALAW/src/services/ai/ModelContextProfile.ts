// src/services/ai/ModelContextProfile.ts
//
// [1.2] Утилита для получения AI-параметров (num_ctx, num_predict, temperature)
// в зависимости от типа задачи и профиля машины.
//
// Использование:
//   import { getContextParams } from '@/services/ai/ModelContextProfile';
//   const params = getContextParams('chat');
//   → { num_ctx: 4096, num_predict: 512, temperature: 0.7 }

import { useOllamaStore, type MachineProfile } from '../../store/ollamaStore';

// ─── Типы задач ──────────────────────────────────────────────────────────────

export type TaskType =
  | 'chat'        // Интерактивный чат Laila (баланс контекста и скорости)
  | 'analyze'     // Анализ документа (нужен большой контекст, точный ответ)
  | 'summarize'   // Суммаризация клиента/дела/апелляции
  | 'rag'         // RAG-ответ (Knowledge, Widget с контекстом)
  | 'classify'    // Классификация документа (короткий вывод, быстро)
  | 'contract'    // Анализ договора (структурированный JSON-вывод)
  | 'risk'        // RiskScanner (список рисков)
  | 'kg_extract'  // Извлечение сущностей для Knowledge Graph (JSON-вывод, средний контекст)
  | 'embed';      // Эмбеддинги (не используется для LLM, для справки)

export interface ContextParams {
  num_ctx: number;          // Размер контекстного окна (токены)
  num_predict: number;      // Максимум генерируемых токенов
  temperature: number;      // Температура (0 = детерминировано, 1 = творчески)
  top_p?: number;           // Nucleus sampling (опционально)
  repeat_penalty?: number;  // Штраф за повторение (1.1–1.3, 1.0 = выключен)
  repeat_last_n?: number;   // Окно для штрафа повторений (токенов назад)
}

// ─── Таблица параметров ──────────────────────────────────────────────────────
//
// Строки: профиль машины
// Столбцы: тип задачи
//
// Логика:
//   - minimal: экономим память, короткие окна, быстрый вывод
//   - light:   умеренные окна, допускаем чуть более длинный вывод
//   - balanced/full: полные окна, качественный вывод

type ProfileTable = Record<TaskType, ContextParams>;

const PROFILES: Record<NonNullable<MachineProfile>, ProfileTable> = {
  minimal: {
    chat:       { num_ctx: 2048, num_predict: 512,  temperature: 0.7, repeat_penalty: 1.15, repeat_last_n: 64 },
    analyze:    { num_ctx: 4096, num_predict: 768,  temperature: 0.2, repeat_penalty: 1.15, repeat_last_n: 64 },
    summarize:  { num_ctx: 2048, num_predict: 512,  temperature: 0.3, repeat_penalty: 1.15, repeat_last_n: 64 },
    rag:        { num_ctx: 2048, num_predict: 512,  temperature: 0.2, repeat_penalty: 1.15, repeat_last_n: 64 },
    classify:   { num_ctx: 1024, num_predict: 150,  temperature: 0.1, repeat_penalty: 1.1,  repeat_last_n: 32 },
    contract:   { num_ctx: 4096, num_predict: 768,  temperature: 0.1, repeat_penalty: 1.15, repeat_last_n: 64 },
    risk:       { num_ctx: 2048, num_predict: 512,  temperature: 0.2, repeat_penalty: 1.15, repeat_last_n: 64 },
    kg_extract: { num_ctx: 2048, num_predict: 800,  temperature: 0.1, repeat_penalty: 1.1,  repeat_last_n: 32 },
    embed:      { num_ctx: 512,  num_predict: 0,    temperature: 0.0 },
  },
  light: {
    chat:       { num_ctx: 4096, num_predict: 1024, temperature: 0.7, repeat_penalty: 1.15, repeat_last_n: 64 },
    analyze:    { num_ctx: 8192, num_predict: 1024, temperature: 0.2, repeat_penalty: 1.15, repeat_last_n: 64 },
    summarize:  { num_ctx: 4096, num_predict: 768,  temperature: 0.3, repeat_penalty: 1.15, repeat_last_n: 64 },
    rag:        { num_ctx: 4096, num_predict: 768,  temperature: 0.2, repeat_penalty: 1.15, repeat_last_n: 64 },
    classify:   { num_ctx: 1024, num_predict: 150,  temperature: 0.1, repeat_penalty: 1.1,  repeat_last_n: 32 },
    contract:   { num_ctx: 8192, num_predict: 768,  temperature: 0.1, repeat_penalty: 1.15, repeat_last_n: 64 },
    risk:       { num_ctx: 4096, num_predict: 512,  temperature: 0.2, repeat_penalty: 1.15, repeat_last_n: 64 },
    kg_extract: { num_ctx: 4096, num_predict: 1200, temperature: 0.1, repeat_penalty: 1.1,  repeat_last_n: 32 },
    embed:      { num_ctx: 512,  num_predict: 0,    temperature: 0.0 },
  },
  balanced: {
    chat:       { num_ctx: 8192,  num_predict: 1024, temperature: 0.7, repeat_penalty: 1.12, repeat_last_n: 64 },
    analyze:    { num_ctx: 16384, num_predict: 2048, temperature: 0.2, repeat_penalty: 1.12, repeat_last_n: 64 },
    summarize:  { num_ctx: 8192,  num_predict: 1024, temperature: 0.3, repeat_penalty: 1.12, repeat_last_n: 64 },
    rag:        { num_ctx: 8192,  num_predict: 768,  temperature: 0.2, repeat_penalty: 1.12, repeat_last_n: 64 },
    classify:   { num_ctx: 2048,  num_predict: 150,  temperature: 0.1, repeat_penalty: 1.1,  repeat_last_n: 32 },
    contract:   { num_ctx: 16384, num_predict: 1536, temperature: 0.1, repeat_penalty: 1.12, repeat_last_n: 64 },
    risk:       { num_ctx: 8192,  num_predict: 768,  temperature: 0.2, repeat_penalty: 1.12, repeat_last_n: 64 },
    kg_extract: { num_ctx: 6144,  num_predict: 1500, temperature: 0.1, repeat_penalty: 1.1,  repeat_last_n: 32 },
    embed:      { num_ctx: 512,   num_predict: 0,    temperature: 0.0 },
  },
  full: {
    chat:       { num_ctx: 16384, num_predict: 2048, temperature: 0.7, repeat_penalty: 1.1,  repeat_last_n: 64 },
    analyze:    { num_ctx: 32768, num_predict: 4096, temperature: 0.2, repeat_penalty: 1.1,  repeat_last_n: 64 },
    summarize:  { num_ctx: 16384, num_predict: 2048, temperature: 0.3, repeat_penalty: 1.1,  repeat_last_n: 64 },
    rag:        { num_ctx: 16384, num_predict: 1024, temperature: 0.2, repeat_penalty: 1.1,  repeat_last_n: 64 },
    classify:   { num_ctx: 2048,  num_predict: 150,  temperature: 0.1, repeat_penalty: 1.1,  repeat_last_n: 32 },
    contract:   { num_ctx: 32768, num_predict: 2048, temperature: 0.1, repeat_penalty: 1.1,  repeat_last_n: 64 },
    risk:       { num_ctx: 16384, num_predict: 1536, temperature: 0.2, repeat_penalty: 1.1,  repeat_last_n: 64 },
    kg_extract: { num_ctx: 8192,  num_predict: 2000, temperature: 0.1, repeat_penalty: 1.1,  repeat_last_n: 32 },
    embed:      { num_ctx: 512,   num_predict: 0,    temperature: 0.0 },
  },
};

// Дефолт при неизвестном профиле (null) — используем 'light' как безопасный средний
const DEFAULT_PROFILE: NonNullable<MachineProfile> = 'light';

// ─── Публичный API ───────────────────────────────────────────────────────────

/**
 * Возвращает AI-параметры для указанного типа задачи
 * с учётом текущего профиля машины из ollamaStore.
 *
 * @example
 *   const { num_ctx, num_predict, temperature } = getContextParams('summarize');
 *   await aiApi.chat(model, messages, false, { num_ctx, num_predict, temperature });
 */
export function getContextParams(taskType: TaskType): ContextParams {
  const profile = useOllamaStore.getState().machineProfile ?? DEFAULT_PROFILE;
  return PROFILES[profile][taskType];
}

/**
 * Проверяет, допускает ли текущий профиль машины
 * запуск ConversationCompressor (сжатие контекста).
 * На minimal — слишком дорого, просто обрезаем историю.
 */
export function canRunCompressor(): boolean {
  const profile = useOllamaStore.getState().machineProfile;
  return profile !== 'minimal';
}

/**
 * Возвращает интервал опроса статуса AI в мс.
 * minimal → 30 сек, full → 10 сек.
 */
export function getAIPollingIntervalMs(): number {
  const profile = useOllamaStore.getState().machineProfile;
  switch (profile) {
    case 'minimal': return 30_000;
    case 'light':   return 20_000;
    case 'balanced':return 15_000;
    case 'full':    return 10_000;
    default:        return 20_000;
  }
}

/**
 * Максимальное число сообщений в STM перед запуском компрессора.
 * Для minimal — не используется (компрессор выключен).
 */
export function getSTMThreshold(): number {
  const profile = useOllamaStore.getState().machineProfile;
  switch (profile) {
    case 'minimal': return Infinity; // никогда не сжимаем
    case 'light':   return 30;
    case 'balanced':return 40;
    case 'full':    return 60;
    default:        return 30;
  }
}
