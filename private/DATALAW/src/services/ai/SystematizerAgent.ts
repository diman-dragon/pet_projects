// src/services/ai/SystematizerAgent.ts
// Агент систематизации документов.
// Классифицирует файл в иерархию Клиент → Дело → Линия → Апелляция
// и извлекает правовые метаданные.

import { OllamaService } from './OllamaService';
import { StorageService } from '../fs/StorageService';
import { businessApi } from '../backend/businessApi';
import { createLogger } from '../logger/Logger';

const log = createLogger('SystematizerAgent');

export interface SystematizationResult {
  client: string;
  client_type?: 'LEGAL' | 'INDIVIDUAL';
  case: string;
  line: string;
  appeal: string;
  metadata?: {
    date?: string | null;
    sender?: string | null;
    recipient?: string | null;
    document_type?: string;
    outgoing_number?: string | null;
    summary?: string;
    suggested_filename?: string;
  };
  confidence: number;
  reasoning: string;
}

// Расширения изображений, которые поддерживает мультимодальный анализ
const IMAGE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'webp']);

// Поддерживаемые PDF (для будущей интеграции — пока только изображения)
// const PDF_EXTENSIONS = new Set(['pdf']);

export class SystematizerAgent {
  /**
   * Анализирует файл и возвращает классификацию в иерархию.
   * Для изображений — мультимодальный анализ через vision.
   * Для остальных форматов — анализ по имени и пути файла.
   */
  static async analyzeDocument(
    fileName: string,
    filePath: string,
  ): Promise<SystematizationResult | null> {
    try {
      // 1. Получаем иерархию для контекста (чтобы подбирать существующие имена)
      const tree = await businessApi.getHierarchyTree();
      const hierarchyContext = tree.map((c) => ({
        client: c.full_name,
        cases: c.cases.map((cs) => ({
          title: cs.title,
          lines: cs.lines.map((l) => l.title),
        })),
      }));

      // 2. Если файл — изображение, конвертируем в base64 для vision-анализа
      let images: string[] | undefined;
      const ext = fileName.split('.').pop()?.toLowerCase() ?? '';

      if (IMAGE_EXTENSIONS.has(ext)) {
        try {
          const bytes = await StorageService.readFile(filePath);
          const base64 = btoa(
            new Uint8Array(bytes).reduce((data, byte) => data + String.fromCharCode(byte), ''),
          );
          images = [base64];
          log.info('Multimodal analysis: image included', { fileName, sizeKb: Math.round(bytes.length / 1024) });
        } catch (imgErr) {
          log.warn('Failed to read image for multimodal analysis, falling back to filename-only', { fileName, imgErr });
        }
      }

      // 3. Формируем пользовательский запрос
      const userMessage = [
        `File Name: ${fileName}`,
        `File Path: ${filePath}`,
        `File Extension: ${ext.toUpperCase() || 'unknown'}`,
        '',
        'Existing Hierarchy Context (match existing names when possible):',
        JSON.stringify(hierarchyContext, null, 2),
      ].join('\n');

      // 4. Запрос к LLM
      const responseText = await OllamaService.chat(
        'SYSTEMATIZE_SYSTEM_PROMPT',
        userMessage,
        [],
        undefined,
        false,
        images,
      );

      // 5. Парсинг JSON
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        log.warn('SystematizerAgent: no JSON found in response', {
          fileName,
          preview: responseText.slice(0, 200),
        });
        return null;
      }

      const result = JSON.parse(jsonMatch[0]) as SystematizationResult;

      // 6. Базовая валидация — все обязательные поля должны быть заполнены
      if (!result.client || !result.case || !result.line || !result.appeal) {
        log.warn('SystematizerAgent: incomplete result, required fields missing', {
          fileName,
          client: result.client,
          case: result.case,
          line: result.line,
          appeal: result.appeal,
        });
        return null;
      }

      log.info('SystematizerAgent: classified', {
        fileName,
        confidence: result.confidence,
        client: result.client,
        case: result.case,
      });

      return result;
    } catch (error) {
      log.error('SystematizerAgent.analyzeDocument failed', { fileName, error });
      return null;
    }
  }
}
