import type { ScannedFile } from '../../types';

export type SortField = 'date' | 'name' | 'size' | 'type';
export type SortDir = 'asc' | 'desc';

export function sortFiles(
  files: ScannedFile[],
  field: SortField,
  dir: SortDir,
  locale: string,
): ScannedFile[] {
  return [...files].sort((a, b) => {
    let cmp = 0;
    switch (field) {
      case 'date':
        cmp = (a.parsedDate?.getTime() ?? 0) - (b.parsedDate?.getTime() ?? 0);
        break;
      case 'name':
        cmp = a.name.localeCompare(b.name, locale);
        break;
      case 'size':
        cmp = a.size - b.size;
        break;
      case 'type':
        cmp = (a.docType ?? 'ZZZ').localeCompare(b.docType ?? 'ZZZ');
        break;
    }
    return dir === 'asc' ? cmp : -cmp;
  });
}
