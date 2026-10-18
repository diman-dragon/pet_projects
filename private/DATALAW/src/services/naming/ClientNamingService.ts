// src/services/naming/ClientNamingService.ts
import { TranslitService } from '../utils/TranslitService';
import { DateFormatService } from '../utils/DateFormatService';

export interface ClientNamingContract {
    /** Когда создавать папку клиента */
    shouldCreateFolder: (client: { full_name: string; client_type: string; status: string }) => boolean;
    /** Когда переименовывать папку клиента */
    shouldRenameFolder: (client: { full_name: string; folder_name: string }) => boolean;
    /** Генерация имени папки */
    generateFolderName: (client: { id: number; full_name: string; client_type: string; status: string }) => string;
}

export const ClientNamingService: any = {
    /**
     * Контракт для клиентов
     */
    contract: {
        shouldCreateFolder: (client: { full_name: string; client_type: string; status: string }) => {
            return client.status === 'ACTIVE' && !!client.full_name.trim();
        },

        shouldRenameFolder: (client: { full_name: string; folder_name: string }) => {
            const expectedName = ClientNamingService.generateFolderName({
                id: 0, // не используется в генерации
                full_name: client.full_name,
                client_type: 'LEGAL', // по умолчанию
                status: 'ACTIVE'
            });
            return client.folder_name !== expectedName;
        },

        generateFolderName: (client: { id: number; full_name: string; client_type: string; status: string }) => {
            const idBlock = DateFormatService.toIdBlock(undefined, new Date());
            const name = ClientNamingService.toClientName(client.full_name, client.client_type);
            return `CL_${idBlock}_${name}_ACTIVE`;
        }
    } as ClientNamingContract,

    /**
     * Генерация имени для клиента
     * IP (Физлица): "Иванов Иван" → "IVANOV_I_I"
     * LE/NO (Юрлица): "АО Вектор" → "AO_VEKTOR"
     */
    toClientName(text: string, clientType: string): string {
        if (clientType === 'INDIVIDUAL') {
            return ClientNamingService.toShortName(text);
        } else {
            return ClientNamingService.toLegalName(text);
        }
    },

    /**
     * IP (Физлица): "Иванов Иван" → "IVANOV_I_I"
     */
    toShortName(text: string): string {
        const parts = String(text || '')
            .trim()
            .split(/\s+/)
            .filter(Boolean);
        if (!parts.length) return '';

        const lastName = TranslitService.transliterate(parts[0]).toUpperCase();
        const initials = parts
            .slice(1)
            .map((p) => {
                const firstChar = p.charAt(0);
                return TranslitService.transliterate(firstChar).toUpperCase();
            })
            .filter(Boolean);

        const result = initials.length ? `${lastName}_${initials.join('_')}` : lastName;
        return result.substring(0, 30);
    },

    /**
     * LE/NO (Юрлица): "АО Вектор" → "AO_VEKTOR"
     */
    toLegalName(text: string): string {
        const str = String(text || '').trim();
        if (!str) return '';

        const result = TranslitService.transliterate(str)
            .toUpperCase()
            .replace(/\s+/g, '_')
            .replace(/[^A-Z0-9_]/g, '')
            .replace(/_+/g, '_')
            .replace(/(^_|_$)/g, '');

        return result.substring(0, 40);
    },
};