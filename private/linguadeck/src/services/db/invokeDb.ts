import { invoke } from "@tauri-apps/api/core";
import { DatabaseService } from "./DatabaseService";

export function invokeDb<T>(command: string, args: Record<string, unknown> = {}) {
  return invoke<T>(command, {
    dbPath: DatabaseService.path,
    ...args
  });
}
