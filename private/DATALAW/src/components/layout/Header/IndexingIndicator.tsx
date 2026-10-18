import React from 'react';
import { Loader2 } from 'lucide-react';
import { useIndexerStore } from '../../../store/indexerStore';

export const IndexingIndicator: React.FC = () => {
    const { status, progress } = useIndexerStore();
    // const { t } = useI18n();  // TODO: add i18n keys when needed

    const isActive = status === 'running' || status === 'checking';

    if (!isActive) return null;

    const progressPct = progress?.filesTotal
        ? Math.min(100, Math.round((progress.filesDone / progress.filesTotal) * 100))
        : progress?.chunksTotal
            ? Math.min(100, Math.round((progress.chunksDone / progress.chunksTotal) * 100))
            : 0;

    const displayName = progress?.currentFile || (status === 'checking' ? 'Проверка...' : 'Индексация...');

    return (
        <div className="flex items-center gap-3 px-3 py-1.5 bg-white/50 rounded-full border border-black/[0.05] shadow-sm animate-in fade-in slide-in-from-top-2 duration-300">
            <div className="relative flex items-center justify-center">
                <Loader2 size={14} className="animate-spin text-[var(--app-accent)]" />
            </div>

            <div className="flex flex-col min-w-[120px]">
                <div className="flex items-center justify-between gap-4">
                    <span className="text-[10px] font-bold text-black/60 uppercase tracking-tight">
                        Индексация
                    </span>
                    {progress && (
                        <span className="text-[10px] font-black text-blue-600">
                            {progress.filesDone}/{progress.filesTotal}
                        </span>
                    )}
                </div>
                <div className="w-full h-1 bg-black/5 rounded-full mt-0.5 overflow-hidden">
                    <div
                        className="h-full bg-[var(--app-accent)] transition-all duration-500"
                        style={{ width: `${progressPct}%` }}
                    />
                </div>
            </div>

            {displayName && (
                <span className="text-[10px] text-black/40 truncate max-w-[120px] font-medium italic">
                    {displayName}
                </span>
            )}
        </div>
    );
};
