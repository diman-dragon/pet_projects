import { useState } from 'react';
import { FileText, FolderOpen, RefreshCw, Sparkles, Calendar, AlertCircle, ChevronLeft, ChevronRight, Loader2, CheckCircle2, XCircle } from 'lucide-react';
import { PageHeader } from '../../components/ui/layout/PageHeader';
import { SegmentedControl } from '../../components/ui/layout/SegmentedControl';
import { Button } from '../../components/ui/form/Button';
import { PageLoader } from '../../components/ui/feedback/PageLoader';
import { Window } from '../../components/ui/layout/Window';
import { useI18n } from '../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../i18n/translations';
import { useAppealDetailsLogic } from './useAppealDetailsLogic';
import { DocumentGrid } from './components/DocumentGrid';
import { DocumentPreviewDrawer } from './components/DocumentPreviewDrawer';
import { AppealTimeline } from './components/AppealTimeline';
import { SummaryView } from './components/SummaryView';
import { StatusBadge } from './components/AppealDetailsPage/StatusBadge';
import { AppealInfoDropdown } from './components/AppealDetailsPage/AppealInfoDropdown';
import type { AppealView, ScannedFile } from './types';

export default function AppealDetailsPage({
  appealId,
  onBack,
}: {
  appealId: number;
  onBack: () => void;
}) {
  const { t } = useI18n();
  const logic = useAppealDetailsLogic(appealId);
  const [view, setView] = useState<AppealView>('documents');
  const [previewFile, setPreviewFile] = useState<ScannedFile | null>(null);
  const [isPreviewExpanded, setIsPreviewExpanded] = useState(false);

  if (logic.loading) return <PageLoader />;

  if (logic.error || !logic.appeal) {
    return (
      <div className="h-full flex items-center justify-center bg-[var(--app-surface-soft)] p-6">
        <Window title={t('common.error')}>
          <div className="p-8 text-center max-w-sm">
            <AlertCircle size={32} className="text-red-500 mx-auto mb-4 opacity-50" />
            <p className="text-[length:var(--text-sm)] font-medium text-black/70 mb-2">
              {logic.error
                ? t(logic.error as TranslationKey) === logic.error
                  ? logic.error
                  : t(logic.error as TranslationKey)
                : t('appeal.error.not_found')}
            </p>
            {logic.errorDetail && (
              <p className="text-[length:var(--text-xs)] font-mono text-red-500/70 mb-6 break-all bg-red-50 rounded-lg p-3 text-left">
                {logic.errorDetail}
              </p>
            )}
            <Button onClick={onBack}>{t('common.back')}</Button>
          </div>
        </Window>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-[var(--app-surface-soft)] overflow-hidden font-sans animation-in fade-in duration-300">
      <PageHeader
        title={logic.appeal.title}
        subtitle={`${logic.appeal.agency_code || t('common.no_code')} · ${t('appeal.type')}`}
        icon={<FileText size={20} className="text-[var(--app-accent)]" />}
        onBack={onBack}
        rightContent={
          <div className="flex items-center gap-2">
            <StatusBadge status={logic.appeal.status ?? 'DRAFT'} />
            <AppealInfoDropdown appeal={logic.appeal} />
            <SegmentedControl
              active={view}
              onChange={(id) => setView(id as AppealView)}
              segments={[
                {
                  id: 'documents',
                  label: t('appeal.tabs.documents'),
                  icon: <FileText size={14} />,
                },
                { id: 'summary', label: t('appeal.tabs.summary'), icon: <Sparkles size={14} /> },
                { id: 'timeline', label: t('appeal.tabs.timeline'), icon: <Calendar size={14} /> },
              ]}
            />
            <Button onClick={logic.handleOpenFolder}>
              <div className="flex items-center gap-1.5">
                <FolderOpen size={14} />
                <span>{t('common.folder')}</span>
              </div>
            </Button>
            <Button onClick={logic.folderScan.rescan}>
              <div className="flex items-center gap-1.5">
                <RefreshCw size={14} className={logic.folderScan.loading ? 'animate-spin' : ''} />
                <span>{logic.folderScan.loading ? '...' : t('common.refresh')}</span>
              </div>
            </Button>

            {/* Chain Summary Status Indicator */}
            {logic.chainSummaryStatus !== 'idle' && (
              <div className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider ${logic.chainSummaryStatus === 'processing' ? 'bg-amber-100 text-amber-700 animate-pulse' :
                  logic.chainSummaryStatus === 'completed' ? 'bg-green-100 text-green-700' :
                    logic.chainSummaryStatus === 'error' ? 'bg-red-100 text-red-700' :
                      'bg-gray-100 text-gray-600'
                }`}>
                {logic.chainSummaryStatus === 'processing' && <Loader2 size={12} className="animate-spin" />}
                {logic.chainSummaryStatus === 'completed' && <CheckCircle2 size={12} />}
                {logic.chainSummaryStatus === 'error' && <XCircle size={12} />}
                <span>
                  {logic.chainSummaryStatus === 'processing' && t('chainSummary.processing')}
                  {logic.chainSummaryStatus === 'completed' && t('chainSummary.completed')}
                  {logic.chainSummaryStatus === 'error' && t('chainSummary.error')}
                </span>
              </div>
            )}
          </div>
        }
      />

      <div className="flex-1 flex overflow-hidden p-3 gap-3">
        <main className="flex-1 min-w-0 overflow-hidden">
          <Window title={t('appeal.materials')} noPadding scrollable={false}>
            <div className="h-full flex flex-col overflow-hidden">
              <div className="flex-1 overflow-y-auto custom-scroll p-4">
                {view === 'documents' && (
                  <DocumentGrid
                    files={logic.folderScan.files}
                    loading={logic.folderScan.loading}
                    error={logic.folderScan.error}
                    lastScanned={logic.folderScan.lastScanned}
                    onRescan={logic.folderScan.rescan}
                    onOpen={logic.handleOpenFile}
                    onPreview={setPreviewFile}
                    docMap={logic.docMap}
                    uploadState={logic.uploadState}
                    pendingRetro={logic.pendingRetro}
                    onUpload={logic.handleUpload}
                    onRetroactive={logic.handleRetroactive}
                    appealId={appealId}
                  />
                )}
                {view === 'summary' && (
                  <SummaryView summary={logic.summary} appealSummary={logic.appeal.summary} />
                )}
                {view === 'timeline' && (
                  <AppealTimeline events={logic.deadlines} onOpenFile={logic.handleOpenFile} />
                )}
              </div>
            </div>
          </Window>
        </main>
        {previewFile && (
          <aside className={`${isPreviewExpanded ? 'w-1/2' : 'w-10'} shrink-0 transition-all duration-300 flex flex-col`}>
            <div className="h-full bg-white border-l border-black/[0.06] flex flex-col">
              {/* Collapse/Expand Toggle */}
              <div className="flex items-center justify-between px-2 py-2 border-b border-black/[0.06] bg-white">
                {isPreviewExpanded ? (
                  <>
                    <span className="text-[9px] font-black uppercase tracking-[0.2em] text-black/25 truncate">
                      {t('common.preview')} — {previewFile.name}
                    </span>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setIsPreviewExpanded(false)}
                        className="p-1.5 hover:bg-black/[0.05] rounded-lg text-black/30 hover:text-black transition-colors"
                        title={t('common.collapse')}
                      >
                        <ChevronRight size={14} />
                      </button>
                      <button
                        onClick={() => setPreviewFile(null)}
                        className="p-1.5 hover:bg-red-50 rounded-lg text-black/30 hover:text-red-500 transition-colors"
                      >
                        ×
                      </button>
                    </div>
                  </>
                ) : (
                  <button
                    onClick={() => setIsPreviewExpanded(true)}
                    className="p-1.5 hover:bg-black/[0.05] rounded-lg text-black/30 hover:text-black transition-colors mx-auto"
                    title={t('common.expand')}
                  >
                    <ChevronLeft size={14} />
                  </button>
                )}
              </div>

              {/* Preview Content */}
              {isPreviewExpanded && (
                <div className="flex-1 overflow-hidden">
                  <DocumentPreviewDrawer
                    file={previewFile}
                    onClose={() => setPreviewFile(null)}
                    onOpen={logic.handleOpenFile}
                    onRename={logic.handleRenameToStandard}
                    dbDoc={logic.docMap.get(previewFile.path) ?? null}
                    embedded
                  />
                </div>
              )}
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
