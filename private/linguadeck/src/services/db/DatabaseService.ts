import Database from "@tauri-apps/plugin-sql";

export class DatabaseService {
  private static instance: Database | null = null;
  private static currentPath: string | null = null;

  static async connect(dbPath: string): Promise<Database> {
    if (this.instance && this.currentPath === dbPath) {
      return this.instance;
    }

    this.instance = await Database.load(`sqlite:${dbPath}`);
    this.currentPath = dbPath;
    await this.instance.execute("PRAGMA foreign_keys = ON;");
    return this.instance;
  }

  static reset() {
    this.instance = null;
    this.currentPath = null;
  }

  static get db(): Database {
    if (!this.instance) {
      throw new Error("Database not initialized");
    }
    return this.instance;
  }

  static get path(): string {
    if (!this.currentPath) {
      throw new Error("Database path not initialized");
    }
    return this.currentPath;
  }
}
