export interface Migration {
  version: number;
  key: 'clients_main' | 'user_system' | 'kb_main' | 'pii_vault' | 'legal_corpus';
  description: string;
  sql: string;
}
