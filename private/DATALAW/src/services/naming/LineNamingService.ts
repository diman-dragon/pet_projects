// src/services/naming/LineNamingService.ts
import { TranslitService } from '../utils/TranslitService';

export interface LineNamingContract {
    /** Когда создавать папку линии */
    shouldCreateFolder: (line: { title: string }) => boolean;
    /** Когда переименовывать папку линии */
    shouldRenameFolder: (line: { title: string; folder_name: string }) => boolean;
    /** Генерация имени папки */
    generateFolderName: (line: { id: number; title: string }) => string;
}

export const LineNamingService: any = {
    /**
     * Контракт для линий
     */
    contract: {
        shouldCreateFolder: (line: { title: string }) => {
            return !!line.title.trim();
        },

        shouldRenameFolder: (line: { title: string; folder_name: string }) => {
            const expectedName = LineNamingService.generateFolderName({
                id: 0, // не используется
                title: line.title
            });
            return line.folder_name !== expectedName;
        },

        generateFolderName: (line: { id: number; title: string }) => {
            const name = LineNamingService.toLineName(line.title);
            return `01_${name}`;
        }
    } as LineNamingContract,

    /**
     * Генерация имени для линии
     */
    toLineName(text: string): string {
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