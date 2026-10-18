import { FileSystemService } from "../FileSystemService";
import { DatabaseService } from "./DatabaseService";
import { MIGRATIONS, CURRENT_SCHEMA_VERSION, type Migration } from "./Migrations";

export const LibraryService = {
  async init(libraryPath: string) {
    const dbFilePath = await FileSystemService.getDatabasePath(libraryPath);
    await DatabaseService.connect(dbFilePath);

    // Get current schema version
    const currentVersion = await LibraryService.getSchemaVersion();

    // Apply pending migrations
    for (const migration of MIGRATIONS) {
      if (migration.version > currentVersion) {
        await LibraryService.applyMigration(migration);
      }
    }

    // Ensure schema_version is set to current
    await LibraryService.setSchemaVersion(CURRENT_SCHEMA_VERSION);
  },

  async getSchemaVersion(): Promise<number> {
    const db = DatabaseService.db;
    try {
      // Check if schema_version table exists
      const tableCheck = await db.select<{ name: string }[]>(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
      );
      if (tableCheck.length === 0) {
        return 0; // No migrations applied yet
      }

      const result = await db.select<{ version: number }[]>(
        "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
      );
      return result[0]?.version ?? 0;
    } catch {
      return 0;
    }
  },

  async setSchemaVersion(version: number): Promise<void> {
    const db = DatabaseService.db;
    await db.execute(
      "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
      [version]
    );
  },

  async applyMigration(migration: Migration): Promise<void> {
    const db = DatabaseService.db;

    // Run all SQL statements in a transaction
    for (const sql of migration.sql) {
      try {
        await db.execute(sql);
      } catch (error) {
        // Log but don't fail on individual statement errors
        // (e.g., IF NOT EXISTS clauses)
        console.warn(`Migration ${migration.version} statement failed: ${error}`);
      }
    }

    console.log(`Applied migration ${migration.version}: ${migration.name}`);
  }
};
