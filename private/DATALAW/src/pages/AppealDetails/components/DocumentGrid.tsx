import { useState, useMemo } from 'react';
import { Search, Sparkles, Upload, RefreshCw, AlertCircle } from 'lucide-react';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { appLanguageToBcp47 } from '../../../i18n/config';
import { businessApi } from '../../../services/backend/businessApi';
import type { ScannedFile } from '../types';
import type { AppealDocument } from '../../../types/entities';
import type { UploadState } from '../useAppealDetailsLogic';
import { SortButton } from './DocumentGrid/SortButton';
import { UploadBanner } from './DocumentGrid/UploadBanner';
import { DocRow } from './DocumentGrid/DocRow';
import { sortFiles, type SortField, type SortDir } from './DocumentGrid/utils';
import { GridStats } from './DocumentGrid/GridStats';

interface DocumentGridProps {
  files: ScannedFile[];
  loading: boolean;
  error: string | null;
  lastScanned: Date | null;
  onRescan: () => void;
  onOpen: (path: string) => void;
  onPreview: (file: ScannedFile) => void;
  docMap: Map<string, AppealDocument>;
  uploadState: UploadState;
  pendingRetro: boolean;
  onUpload: () => void;
  onRetroactive: () => void;
  appealId: number | null;
}

export function DocumentGrid({
  files,
  loading,
  error,
  lastScanned,
  onRescan,
  onOpen,
  onPreview,
  docMap,
  uploadState,
  pendingRetro,
  onUpload,
  onRetroactive,
  appealId,
}: DocumentGridProps) {
  const { t, language } = useI18n();
  const localeTag = appLanguageToBcp47(language);
  const [search, setSearch] = useState('');
  const [sortField, setSortField] = useState<SortField>('date');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [filterType, setFilterType] = useState<string>('');

  const handleReanalyze = async (filePath: string) => {
    if (!appealId) return;
    try {
      await businessApi.reanalyzeDocument(appealId, filePath);
      onRescan();
    } catch (err) {
      console.error('Reanalyze failed:', err);
    }
  };

  const docTypes = useMemo(() => {
    const types = new Set(files.map((f) => f.docType).filter(Boolean) as string[]);
    return Array.from(types).sort();
  }, [files]);

  const filtered = useMemo(() => {
    let result = files;
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      result = result.filter(
        (f) =>
          f.name.toLowerCase().includes(q) ||
          (f.docType ?? '').toLowerCase().includes(q) ||
          (f.description ?? '').toLowerCase().includes(q),
      );
    }
    if (filterType) result = result.filter((f) => f.docType === filterType);
    return sortFiles(result, sortField, sortDir, localeTag);
  }, [files, search, filterType, sortField, sortDir, localeTag]);

  const toggleSort = (field: SortField) => {
    if (sortField === field) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else {
      setSortField(field);
      setSortDir('desc');
    }
  };

  return (
    <div className="flex flex-col h-full">
      {uploadState.active && <UploadBanner state={uploadState} />}

      <div className="flex items-center gap-2 mb-3 shrink-0">
        <div className="relative flex-1">
          <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-black/25" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={t('documentGrid.searchPlaceholder')}
            className="w-full pl-7 pr-3 h-7 rounded-md border border-black/[0.08] bg-black/[0.02]
                                   text-[length:var(--text-xs)] text-black/70 placeholder:text-black/25
                                   focus:outline-none focus:ring-1 focus:ring-black/10 transition-all"
          />
        </div>

        {docTypes.length > 0 && (
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="h-7 px-2 rounded-md border border-black/[0.08] bg-black/[0.02]
                                   text-[length:var(--text-xs)] text-black/60 focus:outline-none cursor-pointer"
          >
            <option value="">{t('documentGrid.allTypes')}</option>
            {docTypes.map((ty) => (
              <option key={ty} value={ty}>
                {ty}
              </option>
            ))}
          </select>
        )}

        <SortButton active={sortField === 'date'} dir={sortDir} onClick={() => toggleSort('date')}>
          {t('documentGrid.sort.date')}
        </SortButton>
        <SortButton active={sortField === 'name'} dir={sortDir} onClick={() => toggleSort('name')}>
          {t('documentGrid.sort.name')}
        </SortButton>
        <SortButton active={sortField === 'size'} dir={sortDir} onClick={() => toggleSort('size')}>
          {t('documentGrid.sort.size')}
        </SortButton>

        {pendingRetro && !uploadState.active && (
          <button
            onClick={onRetroactive}
            title={t('documentGrid.retro.title')}
            className="h-7 flex items-center gap-1.5 px-2.5 rounded-md border border-violet-300 bg-violet-50 text-violet-700 text-[length:var(--text-xs)] font-semibold hover:bg-violet-100 transition-all"
          >
            <Sparkles size={11} />
            {t('documentGrid.retro.button')}
          </button>
        )}

        <button
          onClick={onUpload}
          disabled={uploadState.active}
          title={t('documentGrid.uploadTitle')}
          className="h-7 flex items-center gap-1.5 px-2.5 rounded-md border border-black/[0.08] bg-black text-white text-[length:var(--text-xs)] font-semibold hover:bg-black/80 disabled:opacity-40 transition-all"
        >
          <Upload size={11} />
          {t('documentGrid.upload')}
        </button>

        <button
          onClick={onRescan}
          disabled={loading}
          title={t('documentGrid.refreshTitle')}
          className="h-7 w-7 flex items-center justify-center rounded-md border border-black/[0.08] text-black/35 hover:text-black/70 hover:bg-black/[0.04] transition-all disabled:opacity-40"
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      <GridStats
        filteredCount={filtered.length}
        totalCount={files.length}
        lastScanned={lastScanned}
        localeTag={localeTag}
        docMap={docMap}
        files={files}
      />

      {error && (
        <div className="flex items-center gap-2 mb-2 px-3 py-2 rounded-lg bg-red-50 border border-red-100 shrink-0">
          <AlertCircle size={13} className="text-red-500 shrink-0" />
          <span className="text-[length:var(--text-xs)] text-red-700">{error}</span>
        </div>
      )}

      {loading && files.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="w-7 h-7 border-2 border-black/10 border-t-black/40 rounded-full animate-spin" />
        </div>
      ) : filtered.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-4 py-16">
          <div className="text-center">
            <p className="text-[length:var(--text-sm)] font-semibold text-black/40">
              {files.length > 0
                ? t('documentGrid.empty.noMatches')
                : t('documentGrid.empty.folderEmpty')}
            </p>
            <p className="text-[length:var(--text-xs)] text-black/25 mt-1">
              {files.length > 0
                ? t('documentGrid.empty.hintSearch')
                : t('documentGrid.empty.hintUpload')}
            </p>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto rounded-xl border border-black/[0.06]">
          <table className="w-full border-collapse">
            <thead className="sticky top-0 bg-[var(--app-surface-soft)] z-10 border-b border-black/[0.06]">
              <tr>
                <th className="py-2 pl-4 pr-2 text-left text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/25 w-[80px]">
                  {t('documentGrid.col.type')}
                </th>
                <th className="py-2 px-2 text-left text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/25 w-[52px]">
                  {t('documentGrid.col.ext')}
                </th>
                <th className="py-2 px-2 text-left text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/25">
                  {t('documentGrid.col.nameDesc')}
                </th>
                <th className="py-2 px-2 text-left text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/25 w-[80px]">
                  {t('documentGrid.col.status')}
                </th>
                <th className="py-2 px-2 text-left text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/25 w-[42px]">
                  {t('documentGrid.col.ver')}
                </th>
                <th className="py-2 px-2 text-left text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/25 w-[80px]">
                  {t('documentGrid.col.date')}
                </th>
                <th className="py-2 px-2 text-right text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/25 w-[68px]">
                  {t('documentGrid.col.size')}
                </th>
                <th className="py-2 px-2 text-center text-[length:var(--text-xs)] font-black uppercase tracking-widest text-black/25 w-[32px]">
                  {t('documentGrid.col.std')}
                </th>
                <th className="py-2 pl-2 pr-4 w-[64px]" />
              </tr>
            </thead>
            <tbody>
              {filtered.map((file) => (
                <DocRow
                  key={file.path}
                  file={file}
                  onOpen={onOpen}
                  onPreview={onPreview}
                  onReanalyze={handleReanalyze}
                  dbDoc={docMap.get(file.path)}
                  localeTag={localeTag}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
