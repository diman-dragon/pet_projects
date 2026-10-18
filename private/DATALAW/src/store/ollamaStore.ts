import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  aiApi,
  type OllamaDiagnostics,
  type OllamaInstallResult,
  type OllamaModel,
  type SidecarStatus,
} from '../services/backend/aiApi';
import { EmbeddingService } from '../services/rag/EmbeddingService';
import {
  type MachineProfile,
  determineProfile,
  getProfileConfig,
  getRecommendedModel,
} from '../services/ai/MachineProfile';
import { createLogger } from '../services/logger/Logger';

export type { MachineProfile } from '../services/ai/MachineProfile';

const log = createLogger('OllamaStore');

interface OllamaState {
  sidecarStatus: SidecarStatus;
  models: OllamaModel[];
  diagnostics: OllamaDiagnostics | null;
  selectedModel: string | null;
  machineProfile: MachineProfile;
  isLoading: boolean;
  error: string | null;
  installMessage: string | null;

  // Actions
  refreshStatus: () => Promise<SidecarStatus>;
  refreshModels: () => Promise<void>;
  refreshDiagnostics: () => Promise<void>;
  setSelectedModel: (model: string | null) => void;
  pullModel: (modelName: string) => Promise<void>;
  installOllama: () => Promise<OllamaInstallResult | null>;
  startEngine: () => Promise<void>;
  stopEngine: () => Promise<void>;
  initMachineProfile: () => Promise<void>;
}

/**
 * Выбирает наилучшую модель из доступного списка.
 *
 * Логика приоритета:
 * 1. Если текущая выбранная модель доступна И является рекомендованной — оставляем.
 * 2. Если рекомендованная модель (из MachineProfile) доступна — выбираем её.
 * 3. Если любая gemma-модель доступна — выбираем первую (gemma — дефолт проекта).
 * 4. Fallback: первая модель из списка.
 *
 * Это исправляет баг: persist сохранял 'llama3.2:3b' из старой сессии,
 * и он оставался выбранным даже когда основная модель уже gemma3:4b.
 */
function pickBestModel(
  models: OllamaModel[],
  currentSelected: string | null,
  machineProfile: MachineProfile,
): string {
  if (models.length === 0) return currentSelected ?? '';

  const names = models.map((m) => m.name);

  // 1. Рекомендованная по профилю железа
  const recommended = getRecommendedModel(machineProfile);
  const exactRecommended = names.find((n) => n === recommended);
  if (exactRecommended) return exactRecommended;

  // 2. Частичное совпадение рекомендованной (e.g. 'gemma3:4b' vs 'gemma3:4b-instruct-q4_K_M')
  const baseRecommended = recommended.split(':')[0];
  const partialRecommended = names.find((n) => n.startsWith(baseRecommended));
  if (partialRecommended) return partialRecommended;

  // 3. Любая gemma (приоритет проекта)
  const anyGemma = names.find((n) => n.startsWith('gemma'));
  if (anyGemma) return anyGemma;

  // 4. Текущая, если она есть в списке
  if (currentSelected && names.includes(currentSelected)) return currentSelected;

  // 5. Первая доступная
  return names[0];
}

export const useOllamaStore = create<OllamaState>()(
  persist(
    (set, get) => ({
      sidecarStatus: { running: false },
      models: [],
      diagnostics: null,
      selectedModel: null,
      machineProfile: null,
      isLoading: false,
      error: null,
      installMessage: null,

      refreshStatus: async () => {
        try {
          const status = await aiApi.getStatus();
          set({ sidecarStatus: status, error: null });
          return status;
        } catch (err: any) {
          const status = { running: false, error: err.message };
          set({ sidecarStatus: status, error: err.message });
          return status;
        }
      },

      refreshModels: async () => {
        set({ isLoading: true, error: null });
        try {
          const models = await aiApi.listModels();
          set({ models, isLoading: false });

          if (models.length > 0) {
            const { selectedModel, machineProfile } = get();

            // FIX: всегда пересчитываем лучшую модель при обновлении списка.
            // Старый код: обновлял только если selectedModel отсутствует в списке.
            // Проблема: persist сохранял 'llama3.2:3b', она оставалась в Ollama
            // рядом с gemma → условие !models.some(...) = false → llama оставалась выбранной.
            const best = pickBestModel(models, selectedModel, machineProfile);

            if (best !== selectedModel) {
              log.info(
                `[OllamaStore] Switching model: "${selectedModel}" → "${best}"` +
                  ` (profile: ${machineProfile ?? 'unknown'})`,
              );
              EmbeddingService.clearCache();
              set({ selectedModel: best });
            }
          }
        } catch (err: any) {
          set({ error: err.message, isLoading: false });
        }
      },

      refreshDiagnostics: async () => {
        set({ isLoading: true, error: null });
        try {
          const diagnostics = await aiApi.getHardware();
          set({ diagnostics, isLoading: false });
        } catch (err: any) {
          set({ error: err.message, isLoading: false });
        }
      },

      initMachineProfile: async () => {
        try {
          const hw = await aiApi.getHardware();
          const ramGb = hw.total_memory_gb;
          const profile = determineProfile(ramGb);
          const profileConfig = getProfileConfig(ramGb);
          set({ machineProfile: profile, diagnostics: hw });
          log.info(`[MachineProfile] RAM: ${ramGb}GB, Profile: ${profile}`, profileConfig);

          // FIX: после определения профиля сразу пересчитываем выбранную модель,
          // не ждём следующего refreshModels
          const { models, selectedModel } = get();
          if (models.length > 0) {
            const best = pickBestModel(models, selectedModel, profile);
            if (best !== selectedModel) {
              log.info(`[MachineProfile] Updating model to recommended: "${best}"`);
              EmbeddingService.clearCache();
              set({ selectedModel: best });
            }
          }
        } catch (err) {
          console.warn('[MachineProfile] Failed to detect:', err);
        }
      },

      setSelectedModel: (model) => {
        const prev = get().selectedModel;
        set({ selectedModel: model });
        if (prev !== model) {
          EmbeddingService.clearCache();
        }
      },

      pullModel: async (modelName: string) => {
        set({ isLoading: true, error: null });
        try {
          await aiApi.pullModel(modelName);
          await get().refreshModels();
        } catch (err: any) {
          set({ error: err.message, isLoading: false });
        }
      },

      installOllama: async () => {
        set({ isLoading: true, error: null, installMessage: null });
        try {
          const result = await aiApi.installOllama();
          const diagnostics = await aiApi.getHardware().catch(() => null);
          set({
            diagnostics,
            installMessage: result.message,
            isLoading: false,
          });
          return result;
        } catch (err: any) {
          set({ error: err.message, isLoading: false });
          return null;
        }
      },

      startEngine: async () => {
        set({ isLoading: true, error: null });
        try {
          await aiApi.startEngine();
          await get().refreshStatus();
          set({ isLoading: false });
        } catch (err: any) {
          set({ error: err.message, isLoading: false });
        }
      },

      stopEngine: async () => {
        set({ isLoading: true, error: null });
        try {
          await aiApi.stopEngine();
          await get().refreshStatus();
          set({ isLoading: false });
        } catch (err: any) {
          set({ error: err.message, isLoading: false });
        }
      },
    }),
    {
      name: 'ollama-storage',
      partialize: (state) => ({
        selectedModel: state.selectedModel,
        machineProfile: state.machineProfile,
      }),
    },
  ),
);
