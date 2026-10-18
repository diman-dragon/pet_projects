import React from 'react';
import { CheckCircle2, AlertCircle } from 'lucide-react';
import { useI18n } from '../../../../i18n/AppLanguageProvider';
import type { TranslationKey } from '../../../../i18n/translations';
import type { UploadState } from '../../useAppealDetailsLogic';

interface UploadBannerProps {
  state: UploadState;
}

export const UploadBanner: React.FC<UploadBannerProps> = ({ state }) => {
  const { t } = useI18n();
  const translatedFileName =
    state.fileName.includes('.') || state.fileName.includes('/') || state.fileName.includes('\\')
      ? state.fileName
      : t(state.fileName as TranslationKey) === state.fileName
        ? state.fileName
        : t(state.fileName as TranslationKey);
  const stageLabel: Record<UploadState['stage'], string> = {
    idle: '',
    hashing: t('documentUpload.hashing'),
    ai: t('documentUpload.ai'),
    saving: t('documentUpload.saving'),
    chain: t('documentUpload.chain'),
    done: t('documentUpload.done'),
    error: t('documentUpload.error'),
  };

  const isError = state.stage === 'error';
  const isDone = state.stage === 'done';

  return (
    <div
      className={`flex items-center gap-3 px-4 py-2.5 rounded-xl border text-[length:var(--text-xs)] mb-3
            ${
              isError
                ? 'bg-red-50 border-red-200 text-red-700'
                : isDone
                  ? 'bg-green-50 border-green-200 text-green-700'
                  : 'bg-blue-50 border-blue-200 text-blue-700'
            }`}
    >
      {!isDone && !isError && (
        <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin shrink-0" />
      )}
      {isDone && <CheckCircle2 size={14} className="shrink-0" />}
      {isError && <AlertCircle size={14} className="shrink-0" />}
      <div className="flex-1 min-w-0">
        <span className="font-semibold">{translatedFileName}</span>
        <span className="ml-2 opacity-70">
          {isError
            ? (() => {
                const errorKey = state.error as TranslationKey | undefined;
                return errorKey
                  ? t(errorKey) === errorKey
                    ? errorKey
                    : t(errorKey)
                  : t('documentUpload.error');
              })()
            : stageLabel[state.stage]}
        </span>
        {state.progress && <span className="ml-2 font-mono opacity-60">{state.progress}</span>}
      </div>
      {state.stage === 'ai' && state.aiToken && (
        <span className="text-[length:var(--text-xs)] font-mono opacity-50 truncate max-w-[120px]">
          {state.aiToken}
        </span>
      )}
    </div>
  );
};
