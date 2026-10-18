import {
  X,
  ExternalLink,
  FileEdit,
  AlertCircle,
  FileText,
  CheckCircle2,
  Loader2,
} from 'lucide-react';
import { convertFileSrc } from '@tauri-apps/api/core';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { formatFileSizeWithT } from '../../../i18n/formatFileSize';
import { extColor } from '../config';
import type { ScannedFile } from '../types';
import type { AppealDocument } from '../../../types/entities';
import { DocumentAiSummaryBlock } from './DocumentPreviewDrawer/DocumentAiSummaryBlock';
import { DuplicateWarning } from './DocumentPreviewDrawer/DuplicateWarning';
import {
  useDocumentPreview,
  STATUSES,
  FILE_STATUS_HINT,
} from './DocumentPreviewDrawer/useDocumentPreview';
import { useDocumentContent } from './DocumentPreviewDrawer/useDocumentContent';

interface DocumentPreviewDrawerProps {
  file: ScannedFile;
  onClose: () => void;
  onOpen: (path: string) => void;
  onRename: (file: ScannedFile, newStem: string) => Promise<boolean>;
  dbDoc?: AppealDocument | null;
  embedded?: boolean;
}

export function DocumentPreviewDrawer({
  file,
  onClose,
  onOpen,
  onRename,
  dbDoc,
  embedded,
}: DocumentPreviewDrawerProps) {
  const { t } = useI18n();
  const {
    dateStr,
    setDateStr,
    description,
    setDescription,
    version,
    setVersion,
    status,
    setStatus,
    isSaving,
    saved,
    saveError,
    newName,
    isValid,
    handleRename,
    incrementVersion,
    isValidDescription,
  } = useDocumentPreview(file, onRename);

  const { content, loading, error, type } = useDocumentContent(file.path, file.ext);

  const assetUrl = convertFileSrc(file.path);
  const isImage = type === 'image';
  const isPdf = type === 'pdf';
  const isDocx = type === 'docx';
  const isXlsx = type === 'xlsx';
  const isText = type === 'text';

  return (
    <aside className={`${embedded ? 'w-full h-full' : 'w-[450px] shadow-2xl'} border-l border-black/[0.06] flex flex-col shrink-0 bg-white overflow-hidden transition-all`}>
      {!embedded && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-black/[0.06] shrink-0 bg-white z-10">
          <div className="flex flex-col min-w-0">
            <span className="text-[9px] font-black uppercase tracking-[0.2em] text-black/25">
              {t('documentPreview.title')}
            </span>
            <span className="text-[length:var(--text-xs)] font-bold text-black/70 truncate">
              {file.name}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-black/[0.05] rounded-lg text-black/30 hover:text-black transition-colors"
          >
            <X size={14} />
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto custom-scroll bg-white">
        <div className="relative border-b border-black/[0.05] flex items-center justify-center min-h-[300px] bg-black/[0.01]">
          {loading ? (
            <div className="flex flex-col items-center gap-3 py-12 text-black/20">
              <Loader2 size={32} className="animate-spin" />
              <span className="text-[10px] font-bold uppercase tracking-widest">
                {t('common.loading')}
              </span>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center gap-2 py-12 px-6 text-red-400 text-center">
              <AlertCircle size={32} />
              <span className="text-[10px] font-bold uppercase tracking-tight">
                {t('documentPreview.error')}
              </span>
              <span className="text-[10px] opacity-60 font-mono break-all mt-2">{error}</span>
            </div>
          ) : isImage ? (
            <img
              src={assetUrl}
              alt=""
              className="max-w-full max-h-[400px] object-contain p-4 shadow-sm"
            />
          ) : isPdf ? (
            <iframe
              src={assetUrl}
              className="w-full h-[400px] border-none"
              title={t('documentPreview.title')}
            />
          ) : isDocx && content ? (
            <div className="w-full bg-white p-8 overflow-x-hidden">
              <article
                className="prose-preview text-[13px]"
                dangerouslySetInnerHTML={{ __html: content }}
              />
            </div>
          ) : isXlsx && content ? (
            <div className="w-full overflow-x-auto p-4 bg-white">
              <div className="preview-xlsx-table" dangerouslySetInnerHTML={{ __html: content }} />
            </div>
          ) : isText && content ? (
            <div className="w-full p-6 font-mono text-[11px] leading-relaxed text-black/70 whitespace-pre-wrap bg-white">
              {content}
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 py-12 opacity-30 text-center px-6">
              <FileText size={48} strokeWidth={1.5} />
              <span className="text-[10px] font-black uppercase tracking-widest leading-tight mt-2">
                {file.ext.includes('doc')
                  ? t('documentPreview.wordNoPreview')
                  : t('documentPreview.noPreview')}
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 px-4 py-3 border-b border-black/[0.05] bg-white sticky top-0 z-10">
          <span
            className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border border-black/5 shadow-sm ${extColor(file.ext)}`}
          >
            {file.ext.toUpperCase()}
          </span>
          <span className="text-[length:var(--text-xs)] font-mono text-black/30">
            {formatFileSizeWithT(file.size, t)}
          </span>
          {file.isStandard && <CheckCircle2 size={11} className="text-green-500 ml-auto" />}
        </div>

        {dbDoc?.summary && (
          <div className="pt-3 border-b border-black/[0.05] pb-3 bg-white">
            <DocumentAiSummaryBlock dbDoc={dbDoc} />
          </div>
        )}
        {dbDoc?.duplicate_of_path && (
          <div className={dbDoc?.summary ? '' : 'pt-3'}>
            <DuplicateWarning dbDoc={dbDoc} />
          </div>
        )}

        <div className="px-5 py-6 space-y-4 bg-white">
          <div className="flex items-center gap-2 text-black/30 mb-2 border-b border-black/[0.05] pb-2">
            <FileEdit size={12} />
            <span className="text-[10px] font-black uppercase tracking-[0.25em]">
              {t('documentPreview.rename')}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="text-[10px] font-black uppercase tracking-widest text-black/25 block">
                {t('documentPreview.dateLabel')}
              </label>
              <input
                value={dateStr}
                onChange={(e) => setDateStr(e.target.value.replace(/\D/g, '').slice(0, 6))}
                className={`w-full h-8 px-3 rounded-lg border text-[11px] font-mono focus:outline-none focus:ring-2 focus:ring-[var(--app-accent)]/10 transition-all ${/^\d{6}$/.test(dateStr) ? 'border-black/[0.08] bg-black/[0.01]' : 'border-red-200 bg-red-50'}`}
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-[10px] font-black uppercase tracking-widest text-black/25 block">
                {t('documentPreview.version')}
              </label>
              <div className="flex gap-1.5">
                <input
                  value={version}
                  onChange={(e) => setVersion(e.target.value.toLowerCase())}
                  className={`flex-1 h-8 px-3 rounded-lg border text-[11px] font-mono focus:outline-none focus:ring-2 focus:ring-[var(--app-accent)]/10 transition-all ${/^v\d{2,}$/.test(version) ? 'border-black/[0.08] bg-black/[0.01]' : 'border-red-200 bg-red-50'}`}
                />
                <button
                  type="button"
                  onClick={incrementVersion}
                  className="h-8 px-2.5 rounded-lg border border-black/[0.08] text-[11px] font-bold hover:bg-black/[0.04] transition-all active:scale-95"
                >
                  +1
                </button>
              </div>
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-[10px] font-black uppercase tracking-widest text-black/25 block">
              {t('documentPreview.descriptionLabel')}
            </label>
            <input
              value={description}
              onChange={(e) =>
                setDescription(e.target.value.replace(/\s/g, '_').replace(/[^A-Za-z0-9_]/g, ''))
              }
              className={`w-full h-8 px-3 rounded-lg border text-[11px] font-mono focus:outline-none focus:ring-2 focus:ring-[var(--app-accent)]/10 transition-all ${isValidDescription(description) ? 'border-black/[0.08] bg-black/[0.01]' : 'border-red-200 bg-red-50'}`}
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-[10px] font-black uppercase tracking-widest text-black/25 block">
              {t('documentPreview.status')}
            </label>
            <div className="grid grid-cols-2 gap-1.5">
              {STATUSES.map((s) => (
                <button
                  type="button"
                  key={s}
                  onClick={() => setStatus(s)}
                  className={`py-2 px-3 rounded-xl text-[11px] font-bold border transition-all text-left relative overflow-hidden group ${status === s ? 'bg-black text-white border-black shadow-lg shadow-black/10' : 'bg-white text-black/50 border-black/10 hover:border-black/30'}`}
                >
                  <span className="relative z-10">{s}</span>
                  <span
                    className={`block text-[9px] font-normal opacity-50 truncate relative z-10 ${status === s ? 'text-white/70' : ''}`}
                  >
                    {t(FILE_STATUS_HINT[s])}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-xl bg-black/[0.02] border border-black/[0.06] p-4 group transition-all hover:bg-black/[0.03]">
            <p className="text-[9px] font-black uppercase tracking-[0.2em] text-black/20 mb-2">
              {t('documentPreview.result')}
            </p>
            <p className="text-[11px] font-mono break-all text-black/50 group-hover:text-black/70 transition-colors leading-relaxed">
              {newName}
            </p>
          </div>

          {saveError && (
            <p className="text-[10px] text-red-600 font-bold bg-red-50 p-2.5 rounded-lg border border-red-100 flex items-center gap-2 animate-pulse">
              <AlertCircle size={12} /> {t(saveError as any)}
            </p>
          )}

          <button
            type="button"
            onClick={handleRename}
            disabled={!isValid || isSaving}
            className={`w-full py-3.5 rounded-2xl text-[10px] font-black uppercase tracking-[0.2em] transition-all shadow-xl active:scale-95 ${saved ? 'bg-green-500 text-white shadow-green-500/20' : isValid ? 'bg-black text-white hover:bg-black/80 shadow-black/20' : 'bg-black/[0.06] text-black/25 cursor-not-allowed'}`}
          >
            {saved
              ? t('documentPreview.done')
              : isSaving
                ? t('documentPreview.saving')
                : t('documentPreview.apply')}
          </button>
        </div>
      </div>

      <div className="p-5 border-t border-black/[0.06] shrink-0 bg-white">
        <button
          type="button"
          onClick={() => onOpen(file.path)}
          className="w-full flex items-center justify-center gap-2.5 py-4 rounded-2xl bg-black text-white text-[10px] font-black uppercase tracking-[0.25em] hover:bg-black/80 transition-all active:scale-[0.98] shadow-2xl shadow-black/10"
        >
          <ExternalLink size={14} /> {t('documentPreview.openFile')}
        </button>
      </div>
    </aside>
  );
}
