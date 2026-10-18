import { ConnectionManager } from './ConnectionManager';

export interface ForeignKey {
  from: string;
  to: string;
  table: string;
}

export interface ColumnInfo {
  name: string;
  type: string;
  isPk: boolean;
}

export interface TableStructure {
  name: string;
  columns: ColumnInfo[];
  foreignKeys: ForeignKey[];
}

interface TableInfoRow {
  name: string;
  type: string;
  pk: number;
}

interface ForeignKeyRow {
  from: string;
  to: string;
  table: string;
}

export interface TableStructureWithCount extends TableStructure {
  rowCount: number;
}

export const InspectorService = {
  async getDbStructure(connectionKey: string): Promise<TableStructure[]> {
    const db = ConnectionManager.get(connectionKey);
    const tables = await db.select<{ name: string }[]>(
      "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
    );

    return Promise.all(
      tables.map(async (table) => {
        const info = await db.select<TableInfoRow[]>(`PRAGMA table_info(${table.name})`);
        const fkList = await db.select<ForeignKeyRow[]>(`PRAGMA foreign_key_list(${table.name})`);

        return {
          name: table.name,
          columns: info.map((column) => ({
            name: column.name,
            type: column.type,
            isPk: column.pk === 1,
          })),
          foreignKeys: fkList.map((foreignKey) => ({
            from: foreignKey.from,
            to: foreignKey.to,
            table: foreignKey.table,
          })),
        };
      }),
    );
  },

  async getDbStructureWithCounts(connectionKey: string): Promise<TableStructureWithCount[]> {
    const db = ConnectionManager.get(connectionKey);
    const tables = await this.getDbStructure(connectionKey);

    return Promise.all(
      tables.map(async (table) => {
        try {
          const rows = await db.select<{ count: number }[]>(
            `SELECT COUNT(*) as count FROM ${table.name}`,
          );
          return { ...table, rowCount: rows[0]?.count ?? 0 };
        } catch {
          return { ...table, rowCount: 0 };
        }
      }),
    );
  },
};
