import { useState, useEffect, useCallback } from 'react';
import { RagFacade } from '../../../services/rag/RagFacade';
import { EmbeddingModelService } from '../../../services/ai/EmbeddingModelService';

export type EmbeddingMode = 'real' | 'hash' | 'unavailable';

export interface EmbeddingStatus {
  mode: EmbeddingMode;
  model: string;
  available: boolean;
  checked: boolean;
}

/**
 * [2.7] Хук для определения статуса семантического поиска
 */
export function useEmbeddingStatus(): EmbeddingStatus {
  const [status, setStatus] = useState<EmbeddingStatus>({
    mode: 'unavailable',
    model: EmbeddingModelService.getDefault(),
    available: false,
    checked: false,
  });

  const checkStatus = useCallback(async () => {
    try {
      const model = await EmbeddingModelService.get();
      const isAvailable = await RagFacade.isEmbeddingModelAvailable(model);

      // Проверяем есть ли реальные эмбеддинги в БД
      const currentModelEmbeddings = await RagFacade.getEmbeddingCount(model);
      const hasRealEmbeddings = currentModelEmbeddings > 0;

      let mode: EmbeddingMode;
      if (!isAvailable) {
        mode = 'unavailable';
      } else if (hasRealEmbeddings) {
        mode = 'real';
      } else {
        mode = 'hash';
      }

      setStatus({
        mode,
        model,
        available: isAvailable,
        checked: true,
      });
    } catch {
      setStatus({
        mode: 'unavailable',
        model: EmbeddingModelService.getDefault(),
        available: false,
        checked: true,
      });
    }
  }, []);

  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  return status;
}
