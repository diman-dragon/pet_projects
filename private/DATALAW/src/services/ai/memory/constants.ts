import { getStmMaxMessages, type MachineProfile } from '../MachineProfile';
import { EmbeddingModelService } from '../EmbeddingModelService';

export const MEMORY_DB_KEY = 'user_system';

// [6.x] Система памяти использует ту же embedding модель, что и для документов
// Это избавляет от отдельной зависимости nomic-embed-text
export async function getMemoryEmbedModel(): Promise<string> {
    return EmbeddingModelService.get();
}

// [6.2] Константы STM теперь зависят от профиля машины
export function getMemoryConstants(profile: MachineProfile) {
    const stmMaxMessages = getStmMaxMessages(profile);
    return {
        STM_MAX_MESSAGES: stmMaxMessages,
        STM_MAX_TOKENS: profile === 'minimal' ? 800 : 2000,
        STM_MAX_MESSAGES_THRESHOLD: Math.floor(stmMaxMessages * 1.5), // [6.3] Порог для сжатия
        MTM_MAX_SUMMARIES: 8,
        LTM_MAX_RESULTS: 6,
        SUMMARY_TRIGGER_TOKENS: 1800,
        SUMMARY_SAFE_TAIL_MESSAGES: 8,
        MEMORY_CONTEXT_PRIMARY_WEIGHT: 0.78,
        MEMORY_CONTEXT_SECONDARY_WEIGHT: 0.22,
    };
}

// ВСЕГДА используй getMemoryConstants(profile) — статические константы удалены
// чтобы предотвратить случайное использование без учета профиля машины
