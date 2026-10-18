import type { LibraryDocumentWithStatus } from '../../../../services/rag/knowledgeLibrary/types';

export function filterLibraryDocuments(
  documents: LibraryDocumentWithStatus[],
  selectedFolder: string | null,
  searchQuery: string,
) {
  const normalizedQuery = searchQuery.trim().toLowerCase();
  return documents.filter((doc) => {
    const matchesFolder = selectedFolder ? doc.category === selectedFolder : true;
    const matchesSearch = normalizedQuery
      ? doc.title.toLowerCase().includes(normalizedQuery) ||
        doc.file_path.toLowerCase().includes(normalizedQuery)
      : true;
    return matchesFolder && matchesSearch;
  });
}

export function getFolderCounts(documents: LibraryDocumentWithStatus[], folders: string[]) {
  return folders.reduce<Record<string, number>>((acc, folder) => {
    acc[folder] = documents.filter((doc) => doc.category === folder).length;
    return acc;
  }, {});
}
