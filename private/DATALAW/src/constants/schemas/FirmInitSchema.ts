// src/constants/schemas/FirmInitSchema.ts
//
// Структура папок при инициализации фирмы.
//
// Knowledge_Base — корневая папка базы знаний (RAG).
//   Подпапки = категории документов.
//   IndexerService сканирует их рекурсивно и использует имя папки как category.

export const FIRM_INIT_SCHEMA = {
  root: ['Knowledge_Base', 'Clients', 'Legal_Corpus', 'PII_Vault', 'models'],
  knowledge_base: [
    // 'ECHR', // Европейский суд по правам человека
    'DATA_ROW', //Неотсоритрованные документы. все складываем сюда
    // 'UN_ICC', // ООН, МУС, международные трибуналы
    // 'CIS_Courts', // Суды СНГ (ВС РФ, КС РФ, ВАС и т.д.)
    // 'Arbitration', // Международный и внутренний арбитраж
    // 'Templates', // Шаблоны документов
    // 'Legislation', // Законодательство и нормативные акты
    // 'Doctrine', // Доктрина, комментарии, монографии
    // 'Internal', // Внутренние документы фирмы
  ],
};
