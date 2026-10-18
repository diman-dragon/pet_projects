// src/services/naming/CaseNamingService.ts
import { TranslitService } from '../utils/TranslitService.ts';

export interface CaseNamingContract {
    /** Когда создавать папку дела */
    shouldCreateFolder: (caseItem: { title: string; status: string }) => boolean;
    /** Когда переименовывать папку дела */
    shouldRenameFolder: (caseItem: { title: string; folder_name: string }) => boolean;
    /** Генерация имени папки */
    generateFolderName: (caseItem: { title: string; status: string }) => string;
}

export const CaseNamingService: any = {
    /**
     * Контракт для обычных дел (договоры, консультации)
     */
    contract: {
        shouldCreateFolder: (caseItem: { title: string; status: string }) => {
            return caseItem.status === 'ACTIVE' && !!caseItem.title.trim();
        },

        shouldRenameFolder: (caseItem: { title: string; folder_name: string }) => {
            const expectedName = CaseNamingService.generateFolderName({
                title: caseItem.title,
                status: 'ACTIVE'
            });
            return caseItem.folder_name !== expectedName;
        },

        generateFolderName: (caseItem: { title: string; status: string }) => {
            return CaseNamingService.toCaseName(caseItem.title);
        }
    } as CaseNamingContract,

    /**
     * Контракт для судебных дел (litigation) - с динамическим статусом
     */
    litigationContract: {
        shouldCreateFolder: (caseItem: { title: string; status: string }) => {
            return !!caseItem.title.trim();
        },

        shouldRenameFolder: (caseItem: { title: string; folder_name: string; status: string }) => {
            const expectedName = CaseNamingService.generateLitigationFolderName({
                title: caseItem.title,
                status: caseItem.status
            });
            return caseItem.folder_name !== expectedName;
        },

        generateFolderName: (caseItem: { title: string; status: string }) => {
            return CaseNamingService.generateLitigationFolderName(caseItem);
        }
    } as CaseNamingContract,

    /**
     * Генерация имени папки для судебного дела с динамическим статусом
     */
    generateLitigationFolderName(caseItem: { title: string; status: string }): string {
        return CaseNamingService.toCaseName(caseItem.title);
    },

    /**
     * Генерация имени для дела
     */
    toCaseName(text: string): string {
        const str = String(text || '').trim();
        if (!str) return '';

        const result = TranslitService.transliterate(str)
            .toUpperCase()
            .replace(/\s+/g, '_')
            .replace(/[^A-Z0-9_]/g, '')
            .replace(/_+/g, '_')
            .replace(/(^_|_$)/g, '');

        return result.substring(0, 35);
    },
};