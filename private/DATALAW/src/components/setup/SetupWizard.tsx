/**
 * SetupWizard — FULLY FIXED
 *
 * ИСПРАВЛЕНИЯ:
 * 1. OllamaStartStep между установкой Ollama и скачиванием моделей.
 *    Раньше pull стартовал до запуска сервера → "Ollama unreachable".
 * 2. Шаг N из M теперь корректно отображается (одинарные {var} скобки).
 * 3. Логика показа под-шагов без ошибок приоритета операторов.
 */
import { open } from '@tauri-apps/plugin-dialog';
import { useEffect, useMemo, useState } from 'react';
import { FolderOpen, HardDrive, Loader2, Sparkles, CheckCircle2, Shield } from 'lucide-react';
import { AppInitializer } from '../../services/init/AppInitializer';
import { aiApi } from '../../services/backend/aiApi';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { AI_MODELS, DEFAULT_CHAT_MODEL } from '../../services/ai/config';
import { Button, UiText } from '../ui';
import { Win11StepRow } from './Win11StepRow';
import { OllamaInstallStep } from './OllamaInstallStep';
import { OllamaStartStep } from './OllamaStartStep';
import { MultiModelPullStep } from './MultiModelPullStep';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';

type SetupPhase = 'folder' | 'database' | 'ai' | 'done' | 'error';
type PreCheckState = 'idle' | 'checking' | 'done' | 'error';

function SetupDoneScreen({ onComplete, t }: { onComplete: () => void; t: (k: any, o?: any) => string }) {
  return (
    <div className="flex flex-col items-center gap-6 px-8 py-10 text-center animate-[fadeIn_0.5s_ease-out]">
      <style>{`@keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }`}</style>
      <div className="flex size-20 items-center justify-center rounded-full bg-[var(--app-success)]/15 ring-1 ring-[var(--app-success)]/30">
        <CheckCircle2 size={40} className="text-[var(--app-success)]" strokeWidth={1.5} />
      </div>
      <div className="space-y-2">
        <UiText as="h2" size="lg" weight="semibold" heading>{t('setup.done.title')}</UiText>
        <UiText as="p" size="sm" tone="secondary">{t('setup.done.subtitle')}</UiText>
      </div>
      <div className="flex w-full flex-col gap-2 rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-4 py-3 text-left">
        <div className="flex items-center gap-2 text-[length:var(--text-xs)] text-[var(--app-text-secondary)]">
          <Shield size={14} className="text-[var(--app-success)]" />
          <span>{t('setup.done.privacyConfirm')}</span>
        </div>
      </div>
      <Button variant="primary" size="md" icon={<Sparkles size={14} />} onClick={onComplete} className="w-full">
        {t('setup.done.openApp')}
      </Button>
    </div>
  );
}

export function SetupWizard({ onComplete }: { onComplete: () => void }) {
  const { t } = useI18n();
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const masterPassword = 'DUMMY_PASSWORD';
  const [phase, setPhase] = useState<SetupPhase>('folder');
  const [error, setError] = useState<string | null>(null);
  const [dbDone, setDbDone] = useState(false);

  const [needsOllamaInstall, setNeedsOllamaInstall] = useState(false);
  const [needsOllamaStart, setNeedsOllamaStart] = useState(false);
  const [needsModelPull, setNeedsModelPull] = useState(false);
  const [embedModelPulled, setEmbedModelPulled] = useState(false);
  const [chatModelPulled, setChatModelPulled] = useState(false);
  const [ollamaAlreadyInstalled, setOllamaAlreadyInstalled] = useState(false);
  const [ollamaAlreadyRunning, setOllamaAlreadyRunning] = useState(false);
  const [modelAlreadyPulled, setModelAlreadyPulled] = useState(false);

  // Завершение под-шагов
  const [installDone, setInstallDone] = useState(false);
  const [startDone, setStartDone] = useState(false);

  const chatModel = DEFAULT_CHAT_MODEL;
  const embedModel = AI_MODELS.rag;
  const [preCheckState, setPreCheckState] = useState<PreCheckState>('idle');

  useEffect(() => {
    let cancelled = false;
    setPreCheckState('checking');
    const run = async () => {
      try {
        const [diagnostics, models] = await Promise.all([
          aiApi.getHardware(),
          aiApi.listModels().catch(() => []),
        ]);
        if (cancelled) return;
        const embedBase = embedModel.split(':')[0];
        const hasChatModel = models.some((m) => m.name === chatModel);
        const hasEmbedModel = models.some((m) => m.name.startsWith(embedBase));
        setOllamaAlreadyInstalled(diagnostics.ollama_installed);
        setOllamaAlreadyRunning(diagnostics.ollama_running);
        setChatModelPulled(hasChatModel);
        setEmbedModelPulled(hasEmbedModel);
        setModelAlreadyPulled(hasChatModel && hasEmbedModel);
        setPreCheckState('done');
      } catch {
        if (!cancelled) setPreCheckState('error');
      }
    };
    run();
    return () => { cancelled = true; };
  }, [chatModel, embedModel]);

  const stepIndex = phase === 'folder' ? 1 : phase === 'database' ? 2 : 3;
  const totalSteps = 3;

  const folderStatus: 'pending' | 'loading' | 'done' | 'error' =
    (phase === 'database' || phase === 'ai' || phase === 'done') ? 'done'
    : phase === 'error' ? 'error'
    : selectedPath ? 'done'
    : 'pending';

  const dbStatus: 'pending' | 'loading' | 'done' | 'error' =
    phase === 'database' ? 'loading'
    : dbDone ? 'done'
    : phase === 'error' ? 'error'
    : 'pending';

  const aiStatus = useMemo((): 'pending' | 'loading' | 'done' | 'error' => {
    if (phase === 'done') return 'done';
    if (phase === 'ai') return 'loading';
    if (phase === 'error') return 'error';
    return 'pending';
  }, [phase]);

  const aiStepText = useMemo(() => {
    if (phase === 'done') return t('setup.ai.ready');
    if (ollamaAlreadyInstalled && ollamaAlreadyRunning && modelAlreadyPulled) return t('setup.ai.ready');
    if (phase === 'ai') return t('setup.ai.installing');
    return t('setup.ai.pending');
  }, [modelAlreadyPulled, ollamaAlreadyInstalled, ollamaAlreadyRunning, phase, t]);

  const canStart = Boolean(selectedPath) && phase !== 'database' && phase !== 'ai' && phase !== 'done';

  const handleStart = async () => {
    if (!selectedPath) return;
    try {
      setError(null);
      setPhase('database');
      await AppInitializer.setupWithPassword(selectedPath, masterPassword);
      setDbDone(true);

      let installed = ollamaAlreadyInstalled;
      let running = ollamaAlreadyRunning;
      let hasChatModel = chatModelPulled;
      let hasEmbedModel = embedModelPulled;

      if (preCheckState !== 'done') {
        const [diagnostics, models] = await Promise.all([
          aiApi.getHardware(),
          aiApi.listModels().catch(() => []),
        ]);
        const embedBase = embedModel.split(':')[0];
        installed = diagnostics.ollama_installed;
        running = diagnostics.ollama_running;
        hasChatModel = models.some((m) => m.name === chatModel);
        hasEmbedModel = models.some((m) => m.name.startsWith(embedBase));
        setOllamaAlreadyInstalled(installed);
        setOllamaAlreadyRunning(running);
        setChatModelPulled(hasChatModel);
        setEmbedModelPulled(hasEmbedModel);
        setModelAlreadyPulled(hasChatModel && hasEmbedModel);
      }

      const allModelsReady = hasChatModel && hasEmbedModel;
      const needInstall = !installed;
      // После winget-установки Ollama НЕ запускается автоматически — нужен старт
      const needStart = !running || needInstall;

      setNeedsOllamaInstall(needInstall);
      setNeedsOllamaStart(needStart);
      setNeedsModelPull(!allModelsReady);

      if (!needInstall && !needStart && allModelsReady) {
        setPhase('done');
        return;
      }
      setPhase('ai');
    } catch (err: any) {
      setError(err?.message ?? t('setup.errorMessage'));
      setPhase('error');
    }
  };

  useKeyboardShortcuts({
    onEnter: () => {
      if (canStart) handleStart();
    },
  });

  const handleSelectFolder = async () => {
    const selected = await open({ directory: true, multiple: false, title: t('setup.dialogPickFolder') });
    if (selected && typeof selected === 'string') setSelectedPath(selected);
  };

  // Под-шаги показываются последовательно
  // installPhaseComplete = шаг A завершён (или не нужен)
  const installPhaseComplete = !needsOllamaInstall || installDone;
  // startPhaseComplete = шаг B завершён (или не нужен)
  const startPhaseComplete = !needsOllamaStart || startDone;

  if (phase === 'done') {
    return (
      <div className="w-full overflow-hidden rounded-[var(--app-radius-lg)] border border-[var(--app-border)] bg-[var(--app-panel-bg)] shadow-[var(--app-shadow-md)] backdrop-blur-xl backdrop-saturate-150">
        <SetupDoneScreen onComplete={onComplete} t={t} />
      </div>
    );
  }

  return (
    <div className="w-full overflow-hidden rounded-[var(--app-radius-lg)] border border-[var(--app-border)] bg-[var(--app-panel-bg)] shadow-[var(--app-shadow-md)] backdrop-blur-xl backdrop-saturate-150">

      {/* Шапка с номером шага */}
      <div className="section-divider">
        <UiText as="h1" size="md" weight="semibold" heading>
          {t('setup.title')}
        </UiText>
        {/* FIX: i18next настроен prefix:'{' suffix:'}' → одинарные скобки {var} */}
        <UiText as="p" size="xs" tone="secondary" className="mt-1">
          {t('setup.step.n', { current: String(stepIndex), total: String(totalSteps) })}
        </UiText>
      </div>

      <div className="space-y-5 px-[var(--app-modal-padding-x)] py-[var(--app-modal-padding-y-body)]">

        {/* Хранилище */}
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-xl bg-[var(--app-accent)] text-white shadow-[var(--app-shadow-sm)]">
            <HardDrive size={18} />
          </div>
          <div>
            <UiText as="p" size="sm" weight="semibold">{t('setup.storageTitle')}</UiText>
            <UiText as="p" size="xs" tone="secondary">{t('setup.storageSubtitle')}</UiText>
          </div>
        </div>

        {/* Выбор папки */}
        <div className="flex gap-2">
          <div className="label-mono-id flex h-10 flex-1 items-center overflow-hidden rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-raised)] px-3">
            {selectedPath ?? t('setup.pathPlaceholder')}
          </div>
          <button
            type="button"
            onClick={handleSelectFolder}
            disabled={phase === 'database' || phase === 'ai'}
            className="flex h-10 items-center gap-1.5 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-3.5 text-[length:var(--text-xs)] font-semibold text-[var(--app-text-secondary)] transition-colors hover:text-[var(--app-text)] disabled:opacity-40"
          >
            <FolderOpen size={16} />
            {t('setup.browse')}
          </button>
        </div>

        {/* Инфо о AI */}
        <div className="rounded-lg border border-[var(--app-border)] bg-[var(--app-surface-soft)] px-3 py-3">
          <UiText as="p" size="xs" weight="semibold">{t('setup.ai.autoTitle')}</UiText>
          <UiText as="p" size="xs" tone="muted" className="mt-1">
            {t('setup.ai.autoBody', { embed: embedModel, chat: chatModel })}
          </UiText>
        </div>

        {/* Обзор шагов */}
        <div className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4 space-y-0">
          <Win11StepRow
            icon={folderStatus === 'done' ? '✓' : '○'}
            text={selectedPath ? t('setup.step.folderSelected') : t('setup.step.folderPick')}
            status={folderStatus}
          />
          <Win11StepRow
            icon={dbStatus === 'loading' ? '⟳' : dbStatus === 'done' ? '✓' : dbStatus === 'error' ? '✗' : '○'}
            text={dbDone ? t('setup.step.dbDone') : t('setup.step.database')}
            status={dbStatus}
            indeterminate={dbStatus === 'loading'}
          />
          <Win11StepRow
            icon={aiStatus === 'loading' ? '⟳' : aiStatus === 'done' ? '✓' : aiStatus === 'error' ? '✗' : '○'}
            text={aiStepText}
            status={aiStatus}
          />
        </div>

        {/* AI под-шаги */}
        {phase === 'ai' && (
          <div className="space-y-2 rounded-xl border border-[var(--app-border)] bg-[var(--app-surface-soft)] p-4">

            {/* Шаг A: Установка Ollama */}
            {needsOllamaInstall ? (
              <OllamaInstallStep
                onComplete={() => {
                  setInstallDone(true);
                  setNeedsOllamaInstall(false);
                }}
              />
            ) : (
              <Win11StepRow icon="✓" text={t('setup.ollama.alreadyInstalled')} status="done" />
            )}

            {/* Шаг B: Запуск Ollama (после завершения install-фазы) */}
            {installPhaseComplete && needsOllamaStart && (
              <OllamaStartStep
                alreadyRunning={ollamaAlreadyRunning && !installDone}
                onComplete={() => {
                  setStartDone(true);
                  setNeedsOllamaStart(false);
                  if (!needsModelPull) setPhase('done');
                }}
                onError={(msg) => {
                  setError(msg);
                  setPhase('error');
                }}
              />
            )}

            {/* Шаг C: Скачивание моделей (после завершения start-фазы) */}
            {installPhaseComplete && startPhaseComplete && needsModelPull && (
              <MultiModelPullStep
                chatModel={chatModel}
                onComplete={() => setPhase('done')}
                chatModelPulled={chatModelPulled}
                embedModelPulled={embedModelPulled}
              />
            )}

            {/* Всё готово без pull */}
            {installPhaseComplete && startPhaseComplete && !needsModelPull && (
              <Button variant="primary" size="md" icon={<Sparkles size={14} />} onClick={() => setPhase('done')}>
                {t('setup.done.openApp')}
              </Button>
            )}
          </div>
        )}

        {/* Ошибка */}
        {error && (
          <div className="rounded-lg border border-[var(--app-danger)]/20 bg-[var(--app-danger)]/8 px-3 py-2 text-[length:var(--text-xs)] font-semibold text-[var(--app-danger)]">
            {error}
          </div>
        )}

        {/* Кнопка старта */}
        <Button
          variant={phase === 'error' ? 'danger' : 'primary'}
          size="md"
          disabled={!canStart}
          icon={phase === 'database' ? <Loader2 size={14} className="animate-spin" /> : <HardDrive size={14} />}
          onClick={handleStart}
        >
          {phase === 'database' ? t('setup.step.dbProgress') : phase === 'error' ? t('setup.step.retry') : t('setup.step.start')}
        </Button>

        <UiText as="p" size="xs" tone="muted" className="text-center">
          {t('setup.privacyNote')}
        </UiText>
      </div>
    </div>
  );
}
