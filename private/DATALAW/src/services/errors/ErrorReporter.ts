import { appDataDir, join } from '@tauri-apps/api/path';
import { readTextFile, writeFile, mkdir, exists } from '@tauri-apps/plugin-fs';
import { createLogger } from '../logger/Logger';

export interface ErrorReport {
  id: string;
  source: string;
  message: string;
  stack?: string;
  details?: Record<string, unknown>;
  createdAt: string;
}

const MAX_REPORTS = 50;
const log = createLogger('ErrorReporter');

async function getErrorFilePath(): Promise<string> {
  const dataDir = await appDataDir();
  if (!(await exists(dataDir))) {
    await mkdir(dataDir, { recursive: true });
  }
  return await join(dataDir, 'errors.json');
}

async function safeRead(): Promise<ErrorReport[]> {
  try {
    const filePath = await getErrorFilePath();
    if (!(await exists(filePath))) return [];

    const raw = await readTextFile(filePath);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as ErrorReport[]) : [];
  } catch (error) {
    log.warn('Failed to read error reports from file', error);
    return [];
  }
}

async function safeWrite(reports: ErrorReport[]): Promise<void> {
  try {
    const filePath = await getErrorFilePath();
    const data = JSON.stringify(reports.slice(0, MAX_REPORTS), null, 2);
    await writeFile(filePath, new TextEncoder().encode(data));
  } catch (error) {
    log.warn('Failed to persist error report to file', error);
  }
}

function normalizeError(error: unknown): { message: string; stack?: string } {
  if (error instanceof Error) {
    return { message: error.message, stack: error.stack };
  }

  if (typeof error === 'object' && error !== null) {
    const maybeMessage = (error as { message?: unknown }).message;
    const maybeStack = (error as { stack?: unknown }).stack;
    return {
      message: typeof maybeMessage === 'string' ? maybeMessage : JSON.stringify(error),
      stack: typeof maybeStack === 'string' ? maybeStack : undefined,
    };
  }

  return { message: String(error) };
}

export const ErrorReporter = {
  async capture(
    source: string,
    error: unknown,
    details?: Record<string, unknown>,
  ): Promise<ErrorReport> {
    const normalized = normalizeError(error);
    const report: ErrorReport = {
      id: crypto.randomUUID(),
      source,
      message: normalized.message,
      stack: normalized.stack,
      details,
      createdAt: new Date().toISOString(),
    };

    const current = await safeRead();
    await safeWrite([report, ...current]);
    log.error(`Captured error from ${source}`, report);
    return report;
  },

  async list(): Promise<ErrorReport[]> {
    return await safeRead();
  },

  async clear(): Promise<void> {
    await safeWrite([]);
  },

  installGlobalHandlers(): void {
    if (typeof window === 'undefined') return;

    window.addEventListener('error', (event) => {
      this.capture('window.error', event.error ?? event.message, {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
      });
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.capture('window.unhandledrejection', event.reason);
    });
  },
};
