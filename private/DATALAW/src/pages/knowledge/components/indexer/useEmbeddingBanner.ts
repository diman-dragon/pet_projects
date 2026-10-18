import { useEffect, useState } from 'react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { EmbeddingModelService } from '../../../../services/ai/EmbeddingModelService';
import { aiApi } from '../../../../services/backend/aiApi';
import { EmbeddingService } from '../../../../services/rag/EmbeddingService';
import { RagFacade } from '../../../../services/rag/RagFacade';

export type EmbeddingBannerState =
  | { tone: 'success'; text: string; actionLabel?: never; onAction?: never }
  | { tone: 'warning' | 'info'; text: string; actionLabel: string; onAction: () => Promise<void> | void };

export function useEmbeddingBanner(forceReembed: () => void, statsEmbeddingsCount?: number) {
  const { t } = useI18n();
  const [banner, setBanner] = useState<EmbeddingBannerState | null>(null);
  const [bannerBusy, setBannerBusy] = useState(false);
  const [embeddingModel, setEmbeddingModel] = useState(EmbeddingModelService.getDefault());

  useEffect(() => {
    let alive = true;

    const resolveBanner = async () => {
      const model = await EmbeddingModelService.get();
      if (!alive) {
        return;
      }

      setEmbeddingModel(model);
      const available = await EmbeddingService.isModelAvailable(model);
      const realEmbeddingCount = await RagFacade.getEmbeddingCount(model).catch(() => 0);
      if (!alive) {
        return;
      }

      if (available && realEmbeddingCount > 0) {
        setBanner({
          tone: 'success',
          text: t('knowledge.indexer.banner.ready', { model }),
        });
        return;
      }

      if (!available && realEmbeddingCount === 0) {
        setBanner({
          tone: 'warning',
          text: t('knowledge.indexer.banner.missingModel'),
          actionLabel: t('knowledge.indexer.banner.download'),
          onAction: async () => {
            setBannerBusy(true);
            try {
              await aiApi.pullModelWithProgress(model);
            } finally {
              setBannerBusy(false);
            }
          },
        });
        return;
      }

      if (!available && realEmbeddingCount > 0) {
        setBanner({
          tone: 'info',
          text: t('knowledge.indexer.banner.staleEmbeddings'),
          actionLabel: t('knowledge.indexer.banner.reindex'),
          onAction: () => forceReembed(),
        });
        return;
      }

      setBanner(null);
    };

    void resolveBanner();
    return () => {
      alive = false;
    };
  }, [forceReembed, statsEmbeddingsCount, t]);

  return {
    banner,
    bannerBusy,
    embeddingModel,
  };
}
