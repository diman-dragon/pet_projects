export type EmbeddingPromptStyle = 'nomic' | 'mixedbread' | 'bge' | 'none';

export interface EmbeddingModelDefinition {
  id: string;
  label: string;
  dim: number;
  multilingual: boolean;
  promptStyle: EmbeddingPromptStyle;
}

export const EMBEDDING_MODELS: EmbeddingModelDefinition[] = [
  {
    id: 'bge-m3:latest',
    label: 'BGE-M3 (рекомендуется)',
    dim: 1024,
    multilingual: true,
    promptStyle: 'bge',
  },
  {
    id: 'nomic-embed-text-v2-moe:latest',
    label: 'Nomic Embed v2 MoE',
    dim: 768,
    multilingual: true,
    promptStyle: 'nomic',
  },
  {
    id: 'mxbai-embed-large:latest',
    label: 'mxbai Embed Large',
    dim: 1024,
    multilingual: false,
    promptStyle: 'mixedbread',
  },
  {
    id: 'nomic-embed-text:latest',
    label: 'Nomic Embed v1.5',
    dim: 768,
    multilingual: false,
    promptStyle: 'nomic',
  },
];

// BGE-M3 — многоязычная модель, лучший выбор для русскоязычных правовых документов
// DEFAULT: bge-m3:latest — мультиязычная, 1024 dim, лучшая для юридических текстов
// Другие модели ТОЛЬКО по прямому запросу с обязательной переиндексацией
export const DEFAULT_EMBEDDING_MODEL = 'bge-m3:latest';

export function getEmbeddingModelDefinition(model: string): EmbeddingModelDefinition {
  return (
    EMBEDDING_MODELS.find((item) => item.id === model || model.startsWith(item.id.split(':')[0])) ?? {
      id: model,
      label: model,
      dim: 1024,
      multilingual: true,
      promptStyle: 'none',
    }
  );
}
