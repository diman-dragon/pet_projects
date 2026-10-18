import { invoke } from '@tauri-apps/api/core';
import type { ContextParams } from '../ai/ModelContextProfile';

export interface OllamaModel {
  name: string;
  size: number;
  details?: any;
}

export interface SidecarStatus {
  running: boolean;
  version?: string;
  error?: string;
}

export interface ModelRecommendation {
  name: string;
  label: string;
  profile: string;
  min_memory_gb: number;
  recommended_memory_gb: number;
  rationale: string;
}

export interface OllamaDiagnostics {
  os: string;
  arch: string;
  cpu_brand: string;
  cpu_cores: number;
  total_memory_gb: number;
  available_memory_gb: number;
  ollama_installed: boolean;
  ollama_running: boolean;
  ollama_version?: string | null;
  install_supported: boolean;
  recommended_model?: string | null;
  recommended_models: ModelRecommendation[];
}

export interface OllamaInstallResult {
  installed: boolean;
  method: string;
  message: string;
}

export interface ChatMessage {
  role: string;
  content: string;
  images?: string[];
}

export const aiApi = {
  async getStatus(): Promise<SidecarStatus> {
    return invoke('ai_get_status');
  },

  async listModels(): Promise<OllamaModel[]> {
    return invoke('ai_list_models');
  },

  /**
   * [3.1/3.3] Добавлен опциональный ctxParams (num_ctx, num_predict, temperature).
   * Если не передан — Ollama использует дефолты модели.
   */
  async chat(
    model: string,
    messages: ChatMessage[],
    stream: boolean = false,
    ctxParams?: Partial<ContextParams>,
  ): Promise<ChatMessage> {
    return invoke('ai_chat_completion', {
      request: {
        model,
        messages,
        stream,
        // Передаём только если заданы — Rust игнорирует undefined поля
        ...(ctxParams
          ? {
              options: {
                num_ctx:        ctxParams.num_ctx,
                num_predict:    ctxParams.num_predict,
                temperature:    ctxParams.temperature,
                top_p:          ctxParams.top_p,
                repeat_penalty: ctxParams.repeat_penalty,
                repeat_last_n:  ctxParams.repeat_last_n,
              },
            }
          : {}),
      },
    });
  },

  async pullModel(model: string): Promise<void> {
    return invoke('ai_pull_model', { model });
  },

  /**
   * [5.1] Скачивание с прогресс-событиями.
   * Слушайте 'ai-pull-progress' и 'ai-pull-done' через listen().
   */
  async pullModelWithProgress(model: string): Promise<void> {
    return invoke('ai_pull_model_with_progress', { model });
  },

  async getHardware(): Promise<OllamaDiagnostics> {
    return invoke('ai_get_hardware_info');
  },
  async installOllama(): Promise<OllamaInstallResult> {
    return invoke('ai_install_ollama');
  },

  async startEngine(): Promise<void> {
    return invoke('ai_start_sidecar');
  },
  async stopEngine(): Promise<void> {
    return invoke('ai_stop_sidecar');
  },
  async getModelLibrary(): Promise<any> {
    return invoke('ai_get_model_library');
  },
  async scanModels(): Promise<any> {
    return invoke('ai_scan_models');
  },
  async loadModel(_path: string): Promise<void> {
    return Promise.resolve();
  },
  async unloadModel(): Promise<void> {
    return Promise.resolve();
  },
  async sendHeartbeat(): Promise<void> {
    return Promise.resolve();
  },
};
