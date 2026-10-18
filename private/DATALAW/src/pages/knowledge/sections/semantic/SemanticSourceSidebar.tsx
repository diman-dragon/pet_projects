import type { TranslationKey } from '../../../../i18n/translations';
import type { KnowledgeSourceSelection } from '../../types/source';
import { SemanticSourceCard } from './SemanticSourceCard';
import { SemanticSourceEmpty } from './SemanticSourceEmpty';

export function SemanticSourceSidebar({
  selectedSource,
  t,
}: {
  selectedSource: KnowledgeSourceSelection | null;
  t: (key: TranslationKey, params?: Record<string, string | number>) => string;
}) {
  return (
    <div className="hidden w-[380px] shrink-0 flex-col border-l border-black/[0.06] bg-[var(--app-surface-soft)] xl:flex">
      <div className="border-b border-black/[0.06] px-4 py-3">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-black/40">
          {t('knowledge.semantic.sourcePanelTitle')}
        </p>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        {selectedSource
          ? <SemanticSourceCard selectedSource={selectedSource} t={t} />
          : <SemanticSourceEmpty text={t('knowledge.semantic.sourcePanelEmpty')} />}
      </div>
    </div>
  );
}
