import { AlertTriangle, CheckCircle2, Cpu, Download, Play, RefreshCw, Square } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Button } from '../../../components/ui/form/Button';
import { UiText } from '../../../components/ui/atoms/UiText';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { EmbeddingModelService } from '../../../services/ai/EmbeddingModelService';
import { AI_MODELS } from '../../../services/ai/config';
import { aiApi } from '../../../services/backend/aiApi';
import { useOllamaStore } from '../../../store/ollamaStore';
import { ModelList } from './ollama/ModelList';
import { PullPanel } from './ollama/PullPanel';

export function OllamaSettingsSection() {
  const { t } = useI18n();
  const {
    sidecarStatus,
    models,
    diagnostics,
    selectedModel,
    setSelectedModel,
    isLoading,
    error,
    installMessage,
    refreshStatus,
    refreshModels,
    refreshDiagnostics,
    pullModel,
    installOllama,
    startEngine,
    stopEngine,
  } = useOllamaStore();

  const [modelToPull, setModelToPull] = useState('');
  const [pendingChatModel, setPendingChatModel] = useState<string>(selectedModel || '');
  const [pendingEmbedModel, setPendingEmbedModel] = useState<string>(AI_MODELS.rag);
  const [embedPulling, setEmbedPulling] = useState(false);
  const [embedPullError, setEmbedPullError] = useState<string | null>(null);
  const [embedPullDone, setEmbedPullDone] = useState(false);
  const [embedModelSaving, setEmbedModelSaving] = useState(false);
  const [activeEmbeddingModel, setActiveEmbeddingModel] = useState<string>(AI_MODELS.rag);

  useEffect(() => {
    refreshStatus();
    refreshModels();
    refreshDiagnostics();
    void EmbeddingModelService.get().then((model) => {
      setActiveEmbeddingModel(model);
      setPendingEmbedModel(model);
    });
  }, [refreshDiagnostics, refreshModels, refreshStatus]);

  useEffect(() => {
    setPendingChatModel(selectedModel || '');
  }, [selectedModel]);

  const refreshAll = async () => {
    await refreshStatus();
    await refreshModels();
    await refreshDiagnostics();
  };

  const handlePullModel = async () => {
    if (!modelToPull.trim()) return;
    await pullModel(modelToPull);
    setModelToPull('');
  };

  const handleInstallOllama = async () => {
    await installOllama();
    await refreshAll();
  };

  const handlePullRecommended = async (modelName: string) => {
    setModelToPull(modelName);
    await pullModel(modelName);
  };

  const handleApplyChatModel = async () => {
    if (!pendingChatModel || pendingChatModel === selectedModel) return;
    setSelectedModel(pendingChatModel);
  };

  const handleApplyEmbeddingModel = async () => {
    if (!pendingEmbedModel || pendingEmbedModel === activeEmbeddingModel) return;

    setEmbedModelSaving(true);
    setEmbedPullDone(false);
    setEmbedPullError(null);

    try {
      await EmbeddingModelService.set(pendingEmbedModel);
      setActiveEmbeddingModel(pendingEmbedModel);
    } catch (err: any) {
      setEmbedPullError(err?.message ?? 'Failed to save embedding model');
    } finally {
      setEmbedModelSaving(false);
    }
  };

  const handlePullEmbedModel = async () => {
    setEmbedPulling(true);
    setEmbedPullError(null);
    setEmbedPullDone(false);
    try {
      await aiApi.pullModelWithProgress(activeEmbeddingModel);
      await refreshModels();
      setEmbedPullDone(true);
    } catch (err: any) {
      setEmbedPullError(err?.message ?? 'Ошибка скачивания');
    } finally {
      setEmbedPulling(false);
    }
  };

  const embedModelInstalled = models.some(
    (model) =>
      model.name === activeEmbeddingModel ||
      model.name.startsWith(activeEmbeddingModel.split(':')[0]),
  );

  const translate = (key: string) => t(key as any);

  return (
    <section className="overflow-hidden rounded-[12px] border border-black/[0.08] bg-white shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-black/[0.05] px-6 py-5">
        <div>
          <h2 className="flex items-center gap-2 text-[12px] font-semibold uppercase tracking-[0.22em] text-black/40">
            <Cpu size={14} />
            {t('sidebar.menu.ai_control')} (Ollama)
          </h2>
          <p className="mt-2 max-w-2xl text-[13px] leading-relaxed text-black/60">
            {t('settings.ollama.description' as any)}
          </p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          icon={<RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />}
          onClick={refreshAll}
          disabled={isLoading}
        >
          {t('settings.ollama.refreshStatus' as any)}
        </Button>
      </div>

      <div className="grid gap-6 px-6 py-6 lg:grid-cols-2">
        <div className="space-y-6">
          <div className="rounded-[12px] border border-black/[0.05] bg-black/[0.02] p-5">
            <div className="mb-4 flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-black/40">
                {t('settings.ollama.diagnosticsTitle' as any)}
              </span>
              {diagnostics?.ollama_installed ? (
                <CheckCircle2 size={16} className="text-green-600" />
              ) : (
                <AlertTriangle size={16} className="text-amber-600" />
              )}
            </div>

            {diagnostics && (
              <div className="space-y-3 text-[13px] text-black/70">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <UiText size="xs" tone="muted">
                      {t('settings.ollama.hardware.os' as any)}
                    </UiText>
                    <UiText size="xs" weight="semibold">
                      {diagnostics.os} / {diagnostics.arch}
                    </UiText>
                  </div>
                  <div>
                    <UiText size="xs" tone="muted">
                      {t('settings.ollama.hardware.memory' as any)}
                    </UiText>
                    <UiText size="xs" weight="semibold">
                      {diagnostics.total_memory_gb} GB
                    </UiText>
                  </div>
                </div>
                <div>
                  <UiText size="xs" tone="muted">
                    {t('settings.ollama.hardware.cpu' as any)}
                  </UiText>
                  <UiText size="xs" weight="semibold">
                    {diagnostics.cpu_brand} · {diagnostics.cpu_cores} {t('settings.ollama.hardware.cores' as any)}
                  </UiText>
                </div>
                <div>
                  <UiText size="xs" tone="muted">
                    {t('settings.ollama.installState' as any)}
                  </UiText>
                  <UiText size="xs" weight="semibold">
                    {diagnostics.ollama_installed
                      ? t('settings.ollama.installed' as any)
                      : t('settings.ollama.notInstalled' as any)}
                    {diagnostics.ollama_version ? ` · ${diagnostics.ollama_version}` : ''}
                  </UiText>
                </div>
              </div>
            )}

            {!diagnostics?.ollama_installed && diagnostics?.install_supported && (
              <div className="mt-4 space-y-3">
                <UiText size="xs" tone="secondary">
                  {t('settings.ollama.installHint' as any)}
                </UiText>
                <Button
                  variant="primary"
                  size="sm"
                  icon={<Download size={14} />}
                  onClick={handleInstallOllama}
                  disabled={isLoading}
                >
                  {t('settings.ollama.installAction' as any)}
                </Button>
              </div>
            )}

            {installMessage && (
              <div className="mt-4 rounded-lg border border-green-100 bg-green-50 px-3 py-2.5">
                <UiText size="xs" className="text-green-700">
                  {installMessage}
                </UiText>
              </div>
            )}
          </div>

          <div className="rounded-[12px] border border-black/[0.05] bg-black/[0.02] p-5">
            <div className="mb-4 flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-black/40">
                {t('settings.ollama.serverStatus' as any)}
              </span>
              <div
                className={`h-2.5 w-2.5 rounded-full ${sidecarStatus.running ? 'bg-green-500' : 'bg-red-500'}`}
              />
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <UiText size="xs" tone="secondary">
                  Ollama API
                </UiText>
                <UiText
                  size="xs"
                  weight="semibold"
                  className={sidecarStatus.running ? 'text-green-600' : 'text-red-600'}
                >
                  {sidecarStatus.running
                    ? t('settings.ollama.connected' as any)
                    : t('settings.ollama.unavailable' as any)}
                </UiText>
              </div>
              <div className="flex items-center justify-between">
                <UiText size="xs" tone="secondary">
                  {t('settings.ollama.address' as any)}
                </UiText>
                <UiText size="xs" weight="medium" className="font-mono opacity-60">
                  http://127.0.0.1:11434
                </UiText>
              </div>
              <div className="flex gap-2 pt-2">
                <Button
                  variant="secondary"
                  size="sm"
                  icon={<Play size={14} />}
                  onClick={startEngine}
                  disabled={isLoading || sidecarStatus.running}
                >
                  Start
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  icon={<Square size={14} />}
                  onClick={stopEngine}
                  disabled={isLoading || !sidecarStatus.running}
                >
                  Stop
                </Button>
              </div>
            </div>
            {error && (
              <div className="mt-4 flex items-start gap-2.5 rounded-lg border border-red-100 bg-red-50/50 p-3">
                <AlertTriangle size={14} className="mt-0.5 shrink-0 text-red-500" />
                <UiText size="xs" className="leading-normal text-red-600">
                  {error}
                </UiText>
              </div>
            )}
          </div>

          <ModelList
            activeEmbeddingModel={activeEmbeddingModel}
            embedModelInstalled={embedModelInstalled}
            embedModelSaving={embedModelSaving}
            embedPullDone={embedPullDone}
            embedPullError={embedPullError}
            embedPulling={embedPulling}
            isLoading={isLoading}
            models={models}
            pendingChatModel={pendingChatModel}
            pendingEmbedModel={pendingEmbedModel}
            selectedModel={selectedModel || ''}
            sidecarRunning={sidecarStatus.running}
            setPendingChatModel={setPendingChatModel}
            setPendingEmbedModel={setPendingEmbedModel}
            onApplyChatModel={handleApplyChatModel}
            onApplyEmbeddingModel={handleApplyEmbeddingModel}
            onPullEmbedModel={handlePullEmbedModel}
            t={translate}
          />
        </div>

        <div className="space-y-6">
          <PullPanel
            diagnosticsRunning={Boolean(diagnostics?.ollama_running)}
            isLoading={isLoading}
            modelToPull={modelToPull}
            recommendedModels={diagnostics?.recommended_models ?? []}
            setModelToPull={setModelToPull}
            onPullModel={handlePullModel}
            onPullRecommended={handlePullRecommended}
            t={translate}
          />
        </div>
      </div>
    </section>
  );
}
