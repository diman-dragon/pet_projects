import {
  getAllLibraryDocuments,
  getLibraryDocumentById,
  searchLibraryByTitle,
} from './knowledgeLibrary/documentReads';
import {
  getLibraryCategories,
  getLibraryIndexJobStatus,
} from './knowledgeLibrary/metaReads';
import { scanAndAddKnowledgeFolder } from './knowledgeLibrary/scan';
import type {
  LibraryDocument,
  LibraryDocumentWithStatus,
  LibraryIndexJobStatus,
} from './knowledgeLibrary/types';
import { deleteLibraryDocument, updateLibraryDocument } from './knowledgeLibrary/write';

export type {
  LibraryDocument,
  LibraryDocumentWithStatus,
  LibraryIndexJobStatus,
} from './knowledgeLibrary/types';

export const KnowledgeLibraryService = {
  getAll(): Promise<LibraryDocumentWithStatus[]> {
    return getAllLibraryDocuments();
  },
  getById(id: number): Promise<LibraryDocumentWithStatus | null> {
    return getLibraryDocumentById(id);
  },
  async getByCategory(category: string): Promise<LibraryDocumentWithStatus[]> {
    const all = await getAllLibraryDocuments();
    return category === 'all' ? all : all.filter((doc) => doc.category === category);
  },
  getCategories(): Promise<string[]> {
    return getLibraryCategories();
  },
  update(
    id: number,
    data: { title?: string; category?: string; metadata?: Record<string, unknown> },
  ): Promise<void> {
    return updateLibraryDocument(id, data);
  },
  delete(id: number): Promise<void> {
    return deleteLibraryDocument(id);
  },
  searchByTitle(query: string): Promise<LibraryDocument[]> {
    return searchLibraryByTitle(query);
  },
  getIndexJobStatus(libraryId: number): Promise<LibraryIndexJobStatus | null> {
    return getLibraryIndexJobStatus(libraryId);
  },
  scanAndAddFolder(): Promise<string[]> {
    return scanAndAddKnowledgeFolder();
  },
};
