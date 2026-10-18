import { invokeDb } from "./invokeDb";
import type { AppLogEntry } from "../../types/domain";

type LogLevel = AppLogEntry["level"];

export const LogService = {
  write(level: LogLevel, scope: string, message: string, details?: unknown) {
    const payload = details == null ? null : JSON.stringify(details);
    return invokeDb("log_write", { level, scope, message, details: payload }).catch(() => undefined);
  },

  info(scope: string, message: string, details?: unknown) {
    return this.write("info", scope, message, details);
  },

  warn(scope: string, message: string, details?: unknown) {
    return this.write("warn", scope, message, details);
  },

  error(scope: string, message: string, details?: unknown) {
    return this.write("error", scope, message, details);
  },

  getRecent(limit = 200) {
    return invokeDb<AppLogEntry[]>("log_list", { limit });
  },

  clear() {
    return invokeDb("log_clear");
  }
};
