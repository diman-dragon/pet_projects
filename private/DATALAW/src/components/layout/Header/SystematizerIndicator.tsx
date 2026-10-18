import React from 'react';
import { Loader2, Layers } from 'lucide-react';
import { useSystematizerStore } from '../../../store/systematizerStore';
import { useI18n } from '../../../i18n/AppLanguageProvider';

export const SystematizerIndicator: React.FC = () => {
  const { queue, isProcessing, isApplying, progress, currentFile } = useSystematizerStore();
  const { t } = useI18n();

  const active = isProcessing || isApplying;
  const completedCount = queue.filter(
    (j) => j.status === 'completed' || j.status === 'applied',
  ).length;
  const total = queue.length;

  if (total === 0) return null;

  return (
    <div className="flex items-center gap-3 px-3 py-1.5 bg-white/50 rounded-full border border-black/[0.05] shadow-sm animate-in fade-in slide-in-from-top-2 duration-300">
      <div className="relative flex items-center justify-center">
        {active ? (
          <Loader2 size={14} className="animate-spin text-[var(--app-accent)]" />
        ) : (
          <Layers size={14} className="text-black/40" />
        )}
      </div>

      <div className="flex flex-col min-w-[80px]">
        <div className="flex items-center justify-between gap-4">
          <span className="text-[10px] font-bold text-black/60 uppercase tracking-tight">
            {isApplying ? t('systematizer.status.applying') : t('systematizer.status.processing')}
          </span>
          <span className="text-[10px] font-black text-blue-600">
            {completedCount}/{total}
          </span>
        </div>
        <div className="w-full h-1 bg-black/5 rounded-full mt-0.5 overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ${isApplying ? 'bg-[var(--app-success)]' : 'bg-[var(--app-accent)]'}`}
            style={{ width: `${active ? progress : (completedCount / total) * 100}%` }}
          />
        </div>
      </div>

      {currentFile && active && (
        <span className="text-[10px] text-black/40 truncate max-w-[100px] font-medium italic">
          {currentFile}
        </span>
      )}
    </div>
  );
};
