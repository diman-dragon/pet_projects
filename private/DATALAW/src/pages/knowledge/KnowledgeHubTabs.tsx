import type { ReactNode } from 'react';
import { Brain, Library, Search, Zap } from 'lucide-react';
import type { TranslationKey } from '../../i18n/translations';
import type { KnowledgeTab } from './hooks/useKnowledgeHub.types';

export interface KnowledgeHubTabItem {
  id: KnowledgeTab;
  label: string;
  icon: ReactNode;
  badge?: number;
}

export function buildKnowledgeHubTabs(
  t: (key: TranslationKey) => string,
  libraryDocumentCount: number,
): KnowledgeHubTabItem[] {
  return [
    { id: 'semantic', label: t('knowledge.tabs.semantic'), icon: <Brain size={16} /> },
    { id: 'fulltext', label: t('knowledge.tabs.fulltext'), icon: <Search size={16} /> },
    {
      id: 'library',
      label: t('knowledge.tabs.library'),
      icon: <Library size={16} />,
      badge: libraryDocumentCount,
    },
    { id: 'indexing', label: t('knowledge.tabs.indexing'), icon: <Zap size={16} /> },
  ];
}
