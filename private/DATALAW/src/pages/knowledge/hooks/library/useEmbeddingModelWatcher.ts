import { useEffect, useRef } from 'react';
import { EmbeddingModelService } from '../../../../services/ai/EmbeddingModelService';

export function useEmbeddingModelWatcher(onModelChange: () => Promise<void>) {
  const lastModelRef = useRef('');

  useEffect(() => {
    const checkModelChange = async () => {
      const currentModel = await EmbeddingModelService.get();
      if (lastModelRef.current && lastModelRef.current !== currentModel) {
        await onModelChange();
      }
      lastModelRef.current = currentModel;
    };

    void checkModelChange();
    const interval = setInterval(() => void checkModelChange(), 30_000);
    return () => clearInterval(interval);
  }, [onModelChange]);
}
