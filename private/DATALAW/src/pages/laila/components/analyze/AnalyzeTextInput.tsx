// components/analyze/AnalyzeTextInput.tsx

import { FileText, Loader2, Sparkles } from 'lucide-react';
import { UiText } from '../../../../components/ui/atoms/UiText';
import { useI18n } from '../../../../i18n/AppLanguageProvider';

interface AnalyzeTextInputProps {
  value: string;
  onChange: (v: string) => void;
  onRun: () => void;
  isAnalyzing: boolean;
}

/** Левая колонка: поле ввода текста + кнопка запуска анализа */
export function AnalyzeTextInput({ value, onChange, onRun, isAnalyzing }: AnalyzeTextInputProps) {
  const { t } = useI18n();

  return (
    <div className="w-1/2 flex flex-col border-r border-black/[0.06] p-5 gap-3">
      <div className="flex items-center gap-2">
        <FileText size={13} className="text-black/30" />
        <UiText as="span" size="xs" weight="bold" tone="muted" className="uppercase tracking-wider">
          {t('laila.analyze.documentLabel')}
        </UiText>
      </div>

      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={t('laila.analyze.inputPlaceholder')}
        className="flex-1 text-[length:var(--text-sm)] text-black/70 placeholder:text-black/25 bg-black/[0.02] rounded-xl border border-black/[0.06] p-4 resize-none outline-none focus:border-violet-400/40 focus:ring-2 focus:ring-violet-400/10 leading-[var(--leading-body)] transition-all"
      />

      <button
        onClick={onRun}
        disabled={!value.trim() || isAnalyzing}
        className="h-10 rounded-xl bg-gradient-to-r from-violet-500 to-[#0067c0] text-white text-[length:var(--text-xs)] font-black
          tracking-wide hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-all
          shadow-sm shadow-violet-500/20 flex items-center justify-center gap-2"
      >
        {isAnalyzing ? (
          <>
            <Loader2 size={13} className="animate-spin" />
            {t('laila.analyze.streaming')}
          </>
        ) : (
          <>
            <Sparkles size={13} />
            {t('laila.analyze.runAnalysis')}
          </>
        )}
      </button>
    </div>
  );
}
