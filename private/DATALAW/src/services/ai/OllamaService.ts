import { listen } from '@tauri-apps/api/event';
import { aiApi, type ChatMessage } from '../backend/aiApi';
import { useOllamaStore } from '../../store/ollamaStore';
import { AppLanguageService } from '../config/AppLanguageService';
import { getContextParams } from './ModelContextProfile';
import { getLangInstruction } from './prompts/laila.identity';
import * as prompts from './prompts';

export interface OllamaChatOptions {
  messages: ChatMessage[];
  systemContext?: string;
  onToken?: (token: string) => void;
  signal?: AbortSignal;
  responseLanguage?: string;
}

class OllamaServiceClass {
  private async getActiveModel(): Promise<string> {
    const selectedModel = useOllamaStore.getState().selectedModel;
    if (selectedModel) return selectedModel;

    // Fallback to first available model if none selected
    const models = await aiApi.listModels();
    if (models.length > 0) {
      useOllamaStore.getState().setSelectedModel(models[0].name);
      return models[0].name;
    }

    throw new Error('Ollama: No models available. Please pull a model first.');
  }

  private async buildSystemPrompt(basePrompt: string, responseLanguage?: string): Promise<string> {
    // Если промпт уже содержит языковую инструкцию, не добавляем дубликат
    if (basePrompt.includes('Отвечай на том же языке')) {
      return basePrompt;
    }
    const langInstruction = responseLanguage
      ? getLangInstruction(responseLanguage)
      : await AppLanguageService.getPromptInstruction();
    return `${langInstruction}\n\n${basePrompt}`;
  }

  async chat(
    promptType: keyof typeof prompts,
    userMessage: string,
    history: ChatMessage[] = [],
    systemContext?: string,
    stream: boolean = false,
    images?: string[],
    responseLanguage?: string,
  ): Promise<string> {
    const model = await this.getActiveModel();
    const basePrompt = (prompts as any)[promptType] || '';
    const systemPrompt = await this.buildSystemPrompt(basePrompt, responseLanguage);

    // [6.3] Получаем contextParams для задачи chat
    const ctxParams = getContextParams('chat');

    const messages: ChatMessage[] = [
      { role: 'system', content: systemPrompt },
      ...(systemContext ? [{ role: 'system', content: systemContext }] : []),
      ...history,
      { role: 'user', content: userMessage, images },
    ];

    const response = await aiApi.chat(model, messages, stream, ctxParams);
    return response.content;
  }

  /**
   * [3.1] Streaming chat proxy с поддержкой ModelContextProfile.
   */
  async streamChat(
    promptType: keyof typeof prompts,
    userMessage: string,
    onToken: (t: string) => void,
    history: ChatMessage[] = [],
    systemContext?: string,
    images?: string[],
    responseLanguage?: string,
  ): Promise<string> {
    const unlisten = await listen<string>('ai-chat-token', (event) => {
      onToken(event.payload);
    });

    try {
      const result = await this.chat(
        promptType,
        userMessage,
        history,
        systemContext,
        true,
        images,
        responseLanguage,
      );
      return result;
    } finally {
      unlisten();
    }
  }

  /**
   * [3.1] Dossier: суммаризация клиента через OllamaService (перенос из Rust).
   * Принимает сырой текст досье и возвращает структурированную сводку.
   */
  async summarizeDossier(
    dossierText: string,
    onToken?: (t: string) => void,
  ): Promise<string> {
    const model = await this.getActiveModel();
    const ctxParams = getContextParams('summarize');
    const systemPrompt = await this.buildSystemPrompt(
      (prompts as any)['DOSSIER_SYSTEM_PROMPT'] || '',
    );

    const messages: ChatMessage[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: dossierText },
    ];

    if (onToken) {
      let fullText = '';
      const unlisten = await listen<string>('ai-chat-token', (event) => {
        fullText += event.payload;
        onToken(event.payload);
      });
      try {
        const result = await aiApi.chat(model, messages, true, ctxParams);
        return result.content || fullText;
      } finally {
        unlisten();
      }
    }

    const result = await aiApi.chat(model, messages, false, ctxParams);
    return result.content;
  }
}

export const OllamaService = new OllamaServiceClass();
