import { Sparkles } from 'lucide-react';
import { Button } from '../../../../components/ui/form/Button';
import { UiText } from '../../../../components/ui/atoms/UiText';
import type { ModelRecommendation } from '../../../../services/backend/aiApi';

interface PullPanelProps {
  diagnosticsRunning: boolean;
  isLoading: boolean;
  modelToPull: string;
  recommendedModels: ModelRecommendation[];
  setModelToPull: (value: string) => void;
  onPullModel: () => void;
  onPullRecommended: (modelName: string) => void;
  t: (key: string) => string;
}

export function PullPanel({
  diagnosticsRunning,
  isLoading,
  modelToPull,
  recommendedModels,
  setModelToPull,
  onPullModel,
  onPullRecommended,
  t,
}: PullPanelProps) {
  return (
    <>
      <div className="rounded-[12px] border border-black/[0.05] bg-black/[0.02] p-5">
        <span className="mb-4 block text-[11px] font-semibold uppercase tracking-[0.18em] text-black/40">
          {t('settings.ollama.recommendedTitle')}
        </span>
        <div className="space-y-3">
          {recommendedModels.map((model) => (
            <div key={model.name} className="rounded-xl border border-black/[0.05] bg-white p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <Sparkles size={14} className="text-[var(--app-accent)]" />
                    <UiText size="xs" weight="semibold">
                      {model.label}
                    </UiText>
                  </div>
                  <UiText size="xs" tone="muted" className="mt-1">
                    <code>{model.name}</code> · {model.profile}
                  </UiText>
                </div>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => onPullRecommended(model.name)}
                  disabled={isLoading || !diagnosticsRunning}
                >
                  {t('settings.ollama.useRecommendation')}
                </Button>
              </div>
              <UiText size="xs" tone="secondary" className="mt-3 leading-relaxed">
                {model.rationale}
              </UiText>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-[12px] border border-black/[0.05] bg-black/[0.02] p-5">
        <span className="mb-4 block text-[11px] font-semibold uppercase tracking-[0.18em] text-black/40">
          {t('settings.ollama.pullTitle')}
        </span>
        <div className="space-y-4">
          <div className="flex gap-2">
            <input
              type="text"
              value={modelToPull}
              onChange={(e) => setModelToPull(e.target.value)}
              placeholder={t('settings.ollama.pullPlaceholder')}
              className="flex-1 rounded-xl border border-black/[0.08] bg-white px-4 py-2.5 text-[14px] transition-colors placeholder:text-black/30 hover:border-black/[0.15] focus:outline-none focus:ring-2 focus:ring-[var(--app-accent)]"
            />
            <Button
              onClick={onPullModel}
              disabled={isLoading || !modelToPull.trim() || !diagnosticsRunning}
              variant="primary"
              size="md"
            >
              {t('settings.ollama.pullAction')}
            </Button>
          </div>
          <div className="rounded-xl border border-black/[0.05] bg-white p-4">
            <UiText size="xs" tone="secondary" className="leading-relaxed">
              {t('settings.ollama.pullHelp')}
            </UiText>
            <a
              href="https://ollama.com/library"
              target="_blank"
              rel="noreferrer"
              className="mt-4 inline-block text-[12px] font-medium text-[var(--app-accent)] hover:underline"
            >
              {t('settings.ollama.openLibrary')}
            </a>
          </div>
        </div>
      </div>
    </>
  );
}
