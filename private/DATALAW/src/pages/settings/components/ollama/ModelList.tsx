import { AlertTriangle, CheckCircle2, Database } from 'lucide-react';
import { Button } from '../../../../components/ui/form/Button';
import { UiText } from '../../../../components/ui/atoms/UiText';
import { EMBEDDING_MODELS } from '../../../../services/ai/embeddingModels';
import type { OllamaModel } from '../../../../services/backend/aiApi';

interface ModelListProps {
  activeEmbeddingModel: string;
  embedModelInstalled: boolean;
  embedModelSaving: boolean;
  embedPullDone: boolean;
  embedPullError: string | null;
  embedPulling: boolean;
  isLoading: boolean;
  models: OllamaModel[];
  pendingChatModel: string;
  pendingEmbedModel: string;
  selectedModel: string;
  sidecarRunning: boolean;
  setPendingChatModel: (value: string) => void;
  setPendingEmbedModel: (value: string) => void;
  onApplyChatModel: () => void;
  onApplyEmbeddingModel: () => void;
  onPullEmbedModel: () => void;
  t: (key: string) => string;
}

function isEmbeddingModel(name: string): boolean {
  return EMBEDDING_MODELS.some((model) => name.startsWith(model.id.split(':')[0]));
}

export function ModelList({
  activeEmbeddingModel,
  embedModelInstalled,
  embedModelSaving,
  embedPullDone,
  embedPullError,
  embedPulling,
  isLoading,
  models,
  pendingChatModel,
  pendingEmbedModel,
  selectedModel,
  sidecarRunning,
  setPendingChatModel,
  setPendingEmbedModel,
  onApplyChatModel,
  onApplyEmbeddingModel,
  onPullEmbedModel,
  t,
}: ModelListProps) {
  const chatModels = models.filter((model) => !isEmbeddingModel(model.name));

  return (
    <>
      <div className="rounded-[12px] border border-black/[0.05] bg-black/[0.02] p-5">
        <span className="mb-4 block text-[11px] font-semibold uppercase tracking-[0.18em] text-black/40">
          {t('settings.ollama.activeModel')}
        </span>
        {chatModels.length > 0 ? (
          <div className="space-y-3">
            <div className="flex gap-2">
              <select
                value={pendingChatModel}
                onChange={(e) => setPendingChatModel(e.target.value)}
                className="flex-1 rounded-xl border border-black/[0.08] bg-white p-2.5 text-[14px] font-medium transition-colors hover:border-black/[0.15] focus:outline-none focus:ring-2 focus:ring-[var(--app-accent)]"
              >
                {chatModels.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name} ({(model.size / 1024 / 1024 / 1024).toFixed(2)} GB)
                  </option>
                ))}
              </select>
              <Button
                variant="secondary"
                size="sm"
                onClick={onApplyChatModel}
                disabled={isLoading || pendingChatModel === selectedModel || !pendingChatModel}
              >
                Применить
              </Button>
            </div>
            <UiText size="xs" tone="muted">
              {t('settings.ollama.activeModelHelp')}
            </UiText>
          </div>
        ) : (
          <UiText size="xs" tone="muted" className="italic">
            {t('settings.ollama.modelsEmpty')}
          </UiText>
        )}
      </div>

      <div
        className={`rounded-[12px] border p-5 ${embedModelInstalled ? 'border-green-200 bg-green-50/40' : 'border-amber-200 bg-amber-50/40'}`}
      >
        <div className="mb-4 flex items-center justify-between">
          <span className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-black/40">
            <Database size={13} />
            Семантический поиск (RAG)
          </span>
          {embedModelInstalled ? (
            <CheckCircle2 size={16} className="text-green-600" />
          ) : (
            <AlertTriangle size={16} className="text-amber-600" />
          )}
        </div>

        <div className="mb-4 space-y-2">
          <div className="flex items-center justify-between">
            <UiText size="xs" tone="muted">
              Модель эмбеддингов
            </UiText>
            <UiText size="xs" weight="semibold" className="font-mono">
              {activeEmbeddingModel}
            </UiText>
          </div>
          <div className="flex items-center justify-between">
            <UiText size="xs" tone="muted">
              Статус
            </UiText>
            <UiText
              size="xs"
              weight="semibold"
              className={embedModelInstalled ? 'text-green-600' : 'text-amber-600'}
            >
              {embedModelInstalled ? 'Установлена' : 'Не установлена'}
            </UiText>
          </div>
        </div>

        <div className="mb-4 space-y-2">
          <UiText size="xs" tone="muted">
            Embedding model
          </UiText>
          <div className="flex gap-2">
            <select
              value={pendingEmbedModel}
              onChange={(e) => setPendingEmbedModel(e.target.value)}
              disabled={embedModelSaving}
              className="flex-1 rounded-xl border border-black/[0.08] bg-white p-2.5 text-[14px] font-medium transition-colors hover:border-black/[0.15] focus:outline-none focus:ring-2 focus:ring-[var(--app-accent)] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {EMBEDDING_MODELS.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label} ({model.id})
                </option>
              ))}
            </select>
            <Button
              variant="secondary"
              size="sm"
              onClick={onApplyEmbeddingModel}
              disabled={embedModelSaving || pendingEmbedModel === activeEmbeddingModel}
            >
              Применить
            </Button>
          </div>
          <UiText size="xs" tone="secondary" className="leading-relaxed">
            Смена модели потребует полной переиндексации базы знаний.
          </UiText>
        </div>

        {!embedModelInstalled && (
          <div className="space-y-3">
            <UiText size="xs" tone="secondary" className="leading-relaxed">
              Без этой модели RAG-поиск работает в текстовом режиме (FTS) без семантики.
              Скачайте её для активации интеллектуального поиска по документам.
            </UiText>
            <Button
              variant="primary"
              size="sm"
              onClick={onPullEmbedModel}
              disabled={embedPulling || !sidecarRunning}
            >
              {embedPulling ? 'Скачивание…' : `Скачать ${activeEmbeddingModel}`}
            </Button>
            {embedPullError && (
              <div className="flex items-start gap-2 rounded-lg border border-red-100 bg-red-50 px-3 py-2">
                <AlertTriangle size={13} className="mt-0.5 shrink-0 text-red-500" />
                <UiText size="xs" className="text-red-600">
                  {embedPullError}
                </UiText>
              </div>
            )}
          </div>
        )}

        {embedPullDone && (
          <div className="flex items-center gap-2 rounded-lg border border-green-100 bg-green-50 px-3 py-2">
            <CheckCircle2 size={13} className="shrink-0 text-green-600" />
            <UiText size="xs" className="text-green-700">
              Модель установлена — переиндексируйте базу знаний для активации семантики
            </UiText>
          </div>
        )}

        {embedModelInstalled && (
          <UiText size="xs" tone="secondary" className="leading-relaxed">
            Семантический поиск активен. Для переключения на другую embedding-модель потребуется
            переиндексация всей базы знаний.
          </UiText>
        )}
      </div>
    </>
  );
}
