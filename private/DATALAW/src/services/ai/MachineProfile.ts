/**
 * [1.1] Профили машин по RAM
 * Определяет настройки AI на основе доступной памяти
 */

export type MachineProfile = 'minimal' | 'light' | 'balanced' | 'full' | null;

export interface ProfileConfig {
  profile: MachineProfile;
  num_ctx: number;
  num_predict: number;
  stm_max_messages: number;
  recommended_model: string;
  polling_interval_ms: number;
  embed_batch_size: number;
}

const PROFILE_TABLE: Record<Exclude<MachineProfile, null>, Omit<ProfileConfig, 'profile'>> = {
  minimal: {
    num_ctx: 2048,
    num_predict: 256,
    stm_max_messages: 6,
    recommended_model: 'gemma3:1b',
    polling_interval_ms: 30000,
    embed_batch_size: 1,
  },
  light: {
    num_ctx: 4096,
    num_predict: 512,
    stm_max_messages: 8,
    recommended_model: 'gemma3:1b',
    polling_interval_ms: 20000,
    embed_batch_size: 2,
  },
  balanced: {
    num_ctx: 8192,
    num_predict: 1024,
    stm_max_messages: 10,
    recommended_model: 'gemma3:1b',
    polling_interval_ms: 15000,
    embed_batch_size: 3,
  },
  full: {
    num_ctx: 16384,
    num_predict: 2048,
    stm_max_messages: 14,
    recommended_model: 'gemma3:1b',
    polling_interval_ms: 10000,
    embed_batch_size: 3,
  },
};

/**
 * Определить профиль машины по объему RAM
 * Возвращает null если totalMemoryGb <= 0
 */
export function determineProfile(totalMemoryGb: number): MachineProfile {
  if (totalMemoryGb <= 0) return null;
  if (totalMemoryGb < 8) return 'minimal';
  if (totalMemoryGb < 12) return 'light';
  if (totalMemoryGb < 16) return 'balanced';
  return 'full';
}

/**
 * Получить конфигурацию для профиля
 */
export function getProfileConfig(totalMemoryGb: number): ProfileConfig {
  const profile = determineProfile(totalMemoryGb);
  if (!profile) {
    return {
      profile: null,
      num_ctx: 4096,
      num_predict: 512,
      stm_max_messages: 8,
      recommended_model: 'gemma3:1b',
      polling_interval_ms: 20000,
      embed_batch_size: 1,
    };
  }
  return {
    profile,
    ...PROFILE_TABLE[profile],
  };
}

/**
 * Получить максимальное количество STM сообщений
 */
export function getStmMaxMessages(profile: MachineProfile): number {
  return profile ? PROFILE_TABLE[profile].stm_max_messages : 8;
}

/**
 * Получить рекомендуемую модель для профиля
 */
export function getRecommendedModel(profile: MachineProfile): string {
  return profile ? PROFILE_TABLE[profile].recommended_model : 'gemma3:1b';
}

/**
 * Получить интервал опроса Ollama
 */
export function getPollingInterval(profile: MachineProfile): number {
  return profile ? PROFILE_TABLE[profile].polling_interval_ms : 20000;
}

/**
 * Получить размер батча для эмбеддингов
 */
export function getEmbedBatchSize(profile: MachineProfile): number {
  return profile ? PROFILE_TABLE[profile].embed_batch_size : 1;
}
