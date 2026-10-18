import type { FilterState } from './types';

export const DEFAULT_FILTERS: FilterState = {
  titleQuery: '',
  category: '',
  language: '',
  folder: '',
  fileType: '',
  matchType: 'all',
  minScore: 0,
  topK: 20,
};

export const FILE_TYPES = ['pdf', 'docx', 'txt', 'md', 'html', 'rtf', 'odt', 'xlsx', 'csv'];
export const TOP_K_OPTIONS = [10, 20, 50, 100];
