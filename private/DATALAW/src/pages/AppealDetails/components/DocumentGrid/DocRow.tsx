import React from 'react';
import { FileWarning, CheckCircle2, Copy, Sparkles, Eye, ExternalLink, RefreshCw } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import { formatFileSizeWithT } from '../../../../i18n/formatFileSize';
import type { TranslationKey } from '../../../../i18n/translations';
import {
  formatDateShort,
  extIcon,
  extColor,
  DOC_TYPE_META,
  DEFAULT_DOC_META,
  STATUS_CONFIG,
} from '../../config';
import type { ScannedFile } from '../../types';
import type { AppealDocument } from '../../../../types/entities';

interface DocRowProps {
  file: ScannedFile;
  onOpen: (path: string) => void;
  onPreview: (f: ScannedFile) => void;
  onReanalyze?: (filePath: string) => void;
  dbDoc?: AppealDocument | null;
  localeTag: string;
}

export const DocRow: React.FC<DocRowProps> = ({
  file,
  onOpen,
  onPreview,
  onReanalyze,
  dbDoc,
  localeTag,
}) => {
  const { t } = useI18n();
  const meta = file.docType ? DOC_TYPE_META[file.docType] ?? DEFAULT_DOC_META : DEFAULT_DOC_META;
  const hasSummary = !!dbDoc?.summary;
  const isDuplicate = !!dbDoc?.duplicate_of_path;
  const isPlaceholder = dbDoc?.summary === t('documentGrid.aiPlaceholder' as any);

  const statusCfg = file.status ? STATUS_CONFIG[file.status] : null;
  const statusKey = file.status ? (`status.${file.status.toLowerCase()}` as TranslationKey) : null;
  const statusText =
    statusCfg && statusKey
      ? (() => {
          const tr = t(statusKey);
          return tr === statusKey ? file.status : tr;
        })()
      : file.status;

  return (
    <tr className="group border-b border-black/[0.04] hover:bg-black/[0.02] transition-colors">
      <td className="py-2 pl-4 pr-2 w-[80px]">
        {file.docType ? (
          <span
            className={`text-[length:var(--text-xs)] font-black uppercase tracking-wider px-1.5 py-0.5 rounded-md whitespace-nowrap ${meta.color}`}
          >
            {file.docType}
          </span>
        ) : (
          <span className="text-[length:var(--text-xs)] text-black/20">—</span>
        )}
      </td>

      <td className="py-2 px-2 w-[52px]">
        <span
          className={`text-[length:var(--text-xs)] font-mono font-bold px-1.5 py-0.5 rounded-md ${extColor(file.ext)}`}
        >
          {file.ext || '—'}
        </span>
      </td>

      <td className="py-2 px-2 max-w-0 w-full">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-[length:var(--text-md)] leading-none shrink-0 select-none">
            {extIcon(file.ext)}
          </span>
          <div className="min-w-0 flex-1">
            {file.isStandard ? (
              <>
                <p className="text-[length:var(--text-xs)] font-semibold text-black/80 truncate leading-tight">
                  {file.description ?? file.name}
                </p>
                <p className="text-[length:var(--text-xs)] font-mono text-black/25 truncate">
                  {file.name}
                </p>
              </>
            ) : (
              <div className="flex items-center gap-1.5 min-w-0">
                <FileWarning size={10} className="text-amber-400 shrink-0" />
                <p className="text-[length:var(--text-xs)] font-medium text-black/60 truncate">
                  {file.name}
                </p>
              </div>
            )}
          </div>
          {hasSummary && (
            <div title={dbDoc?.summary ?? ''} className="shrink-0">
              <Sparkles size={10} className={isPlaceholder ? 'text-amber-400' : 'text-violet-400'} />
            </div>
          )}
          {isPlaceholder && onReanalyze && (
            <button
              onClick={() => onReanalyze(file.path)}
              title={t('documentGrid.reanalyzeTitle' as any)}
              className="shrink-0 w-5 h-5 flex items-center justify-center rounded-md text-amber-500 hover:text-amber-600 hover:bg-amber-50 transition-all"
            >
              <RefreshCw size={12} />
            </button>
          )}
          {isDuplicate && (
            <div
              title={t('documentUpload.duplicateTitle', { path: dbDoc?.duplicate_of_path ?? '' })}
              className="shrink-0"
            >
              <Copy size={10} className="text-amber-500" />
            </div>
          )}
        </div>
      </td>

      <td className="py-2 px-2 w-[80px]">
        {statusCfg ? (
          <span
            className={`text-[length:var(--text-xs)] font-black uppercase tracking-wider px-1.5 py-0.5 rounded-md whitespace-nowrap ${statusCfg.color}`}
          >
            {statusText}
          </span>
        ) : file.status ? (
          <span className="text-[length:var(--text-xs)] font-black uppercase tracking-wider px-1.5 py-0.5 rounded-md bg-black/[0.05] text-black/50">
            {file.status}
          </span>
        ) : (
          <span className="text-[length:var(--text-xs)] text-black/20">—</span>
        )}
      </td>

      <td className="py-2 px-2 w-[42px]">
        <span className="text-[length:var(--text-xs)] font-mono text-black/30">
          {file.version ?? '—'}
        </span>
      </td>

      <td className="py-2 px-2 w-[80px]">
        <span className="text-[length:var(--text-xs)] font-mono text-black/40 whitespace-nowrap">
          {file.parsedDate ? formatDateShort(file.parsedDate, localeTag) : '—'}
        </span>
      </td>

      <td className="py-2 px-2 w-[68px] text-right">
        <span className="text-[length:var(--text-xs)] font-mono text-black/30">
          {file.size > 0 ? formatFileSizeWithT(file.size, t) : '—'}
        </span>
      </td>

      <td className="py-2 px-2 w-[32px] text-center">
        {file.isStandard ? (
          <CheckCircle2 size={12} className="text-green-500 mx-auto" />
        ) : (
          <span className="text-[length:var(--text-xs)] text-black/15">·</span>
        )}
      </td>

      <td className="py-2 pl-2 pr-4 w-[64px]">
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => onPreview(file)}
            title={t('documentGrid.previewTitle')}
            className="w-6 h-6 flex items-center justify-center rounded-md text-black/35 hover:text-black hover:bg-black/[0.07] transition-all"
          >
            <Eye size={13} />
          </button>
          <button
            onClick={() => onOpen(file.path)}
            title={t('documentGrid.openTitle')}
            className="w-6 h-6 flex items-center justify-center rounded-md text-black/35 hover:text-black hover:bg-black/[0.07] transition-all"
          >
            <ExternalLink size={12} />
          </button>
        </div>
      </td>
    </tr>
  );
};
