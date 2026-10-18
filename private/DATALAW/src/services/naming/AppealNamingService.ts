// src/services/naming/AppealNamingService.ts
import { TranslitService } from '../utils/TranslitService.ts';
import { DateFormatService } from '../utils/DateFormatService.ts';

export interface AppealNamingContract {
    /** Когда создавать папку обращения */
    shouldCreateFolder: (appeal: { title: string; status: string; appeal_date?: string }) => boolean;
    /** Когда переименовывать папку обращения */
    shouldRenameFolder: (appeal: { title: string; folder_name: string; appeal_date?: string }) => boolean;
    /** Генерация имени папки */
    generateFolderName: (appeal: { id: number; title: string; status: string; appeal_date?: string }) => string;
}

export const AppealNamingService: any = {
    /**
     * Контракт для обращений
     */
    contract: {
        shouldCreateFolder: (appeal: { title: string; status: string; appeal_date?: string }) => {
            return !!appeal.title.trim();
        },

        shouldRenameFolder: (appeal: { title: string; folder_name: string; appeal_date?: string }) => {
            const expectedName = AppealNamingService.generateFolderName({
                id: 0, // не используется
                title: appeal.title,
                status: 'OPEN',
                appeal_date: appeal.appeal_date
            });
            return appeal.folder_name !== expectedName;
        },

        generateFolderName: (appeal: { id: number; title: string; status: string; appeal_date?: string }) => {
            const dateBlock = appeal.appeal_date ? DateFormatService.toShortDate(new Date(appeal.appeal_date)) : '000000';
            const idBlock = DateFormatService.toIdBlock(undefined, new Date());
            const name = AppealNamingService.toAppealName(appeal.title);
            return `${dateBlock}_${idBlock}_01_DOC_${name}`;
        }
    } as AppealNamingContract,

    /**
     * Генерация имени для обращения
     */
    toAppealName(text: string): string {
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