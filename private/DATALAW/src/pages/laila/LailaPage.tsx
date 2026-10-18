// LailaPage.tsx
// [3.3] Оптимизация: React.lazy для AnalyzePanel и StatusPanel

import { useMemo, lazy, Suspense } from 'react';
import { Brain, MessageSquare, FileSearch, Activity } from 'lucide-react';
import { PageBase } from '../../shared/ui/PageBase';
import { Badge } from '../../components/ui';
import { OllamaModelSelector } from '../../components/ai/OllamaModelSelector';
import { useI18n } from '../../i18n/AppLanguageProvider';
import { useLailaLogic } from './useLailaLogic';
import { LailaOrb } from './components/orb/LailaOrb';
import { ChatPanel } from './components/chat/ChatPanel';
import { LailaMode } from './types';

// [3.3] Ленивая загрузка тяжёлых панелей
const AnalyzePanel = lazy(() =>
  import('./components/analyze/AnalyzePanel').then((m) => ({ default: m.AnalyzePanel })),
);
const StatusPanel = lazy(() =>
  import('./components/status/StatusPanel').then((m) => ({ default: m.StatusPanel })),
);

function PanelFallback() {
  return (
    <div className="flex flex-1 items-center justify-center text-[var(--app-text-muted)] text-xs">
      <span className="animate-pulse">Загрузка…</span>
    </div>
  );
}

export default function LailaPage() {
  const logic = useLailaLogic();
  const { t } = useI18n();

  const tabs = useMemo(
    () =>
      [
        { id: 'chat' as const, label: t('laila.tab.chat'), icon: <MessageSquare size={13} /> },
        { id: 'analyze' as const, label: t('laila.tab.analyze'), icon: <FileSearch size={13} /> },
        { id: 'status' as const, label: t('laila.tab.status'), icon: <Activity size={13} /> },
      ] as const,
    [t],
  );

  return (
    <PageBase
      title={t('laila.pageTitle')}
      subtitle={t('laila.subtitle')}
      icon={<Brain size={22} strokeWidth={1.8} />}
      activeTab={logic.mode}
      onTabChange={(id) => logic.setMode(id as LailaMode)}
      tabs={[...tabs]}
      extra={
        <div className="flex items-center justify-between px-6 py-2 border-t border-black/[0.03] bg-gray-50/50">
          <div className="flex items-center gap-3">
            <LailaOrb size="sm" isActive={logic.isThinking || logic.isAnalyzing} />
            <span className="text-[length:var(--text-xs)] font-bold text-black/40 uppercase tracking-widest">
              {logic.aiReady === false
                ? t('laila.footer.localmindOffline')
                : t('laila.footer.systemActive')}
            </span>
            <OllamaModelSelector className="hidden lg:flex" />
          </div>
          <div className="flex items-center gap-2">
            <Badge label={t('laila.badge.alpha')} variant="violet" />
            <Badge
              label={logic.aiReady === false ? t('laila.badge.offline') : t('laila.badge.local')}
              variant={logic.aiReady === false ? 'red' : 'gray'}
            />
          </div>
        </div>
      }
      noPadding
    >
      <div className="flex-1 overflow-hidden h-full flex flex-col">
        {logic.mode === 'chat' && (
          <ChatPanel
            messages={logic.messages}
            input={logic.input}
            onInputChange={logic.setInput}
            onSend={logic.sendMessage}
            onClear={logic.clearChat}
            onStop={logic.stopGeneration}
            isThinking={logic.isThinking}
            bottomRef={logic.bottomRef as any}
            sessionId={logic.sessionId}
            onLoadSession={logic.loadSession}
          />
        )}

        {logic.mode === 'analyze' && (
          <Suspense fallback={<PanelFallback />}>
            <AnalyzePanel
              text={logic.analysisText}
              onTextChange={logic.setAnalysisText}
              onRun={logic.runAnalysis}
              isAnalyzing={logic.isAnalyzing}
              result={logic.analysisResult}
              streamingText={logic.streamingText}
              onClear={logic.clearAnalysis}
            />
          </Suspense>
        )}

        {logic.mode === 'status' && (
          <Suspense fallback={<PanelFallback />}>
            <StatusPanel />
          </Suspense>
        )}
      </div>
    </PageBase>
  );
}
