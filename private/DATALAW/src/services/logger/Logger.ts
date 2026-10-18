export interface AppLogger {
  info(...args: unknown[]): void;
  warn(...args: unknown[]): void;
  error(...args: unknown[]): void;
  debug(...args: unknown[]): void;
  critical(...args: unknown[]): void;
}

type LogEntry = {
  level: string;
  tag: string;
  message: string;
  details?: string | null;
  timestamp: string;
};

type DbHandler = (entry: LogEntry) => Promise<void>;

let activeDbHandler: DbHandler | null = null;
const pendingLogs: LogEntry[] = [];
let isProcessing = false;

function stringifyArg(arg: unknown): string {
  if (typeof arg === 'string') return arg;
  try {
    return JSON.stringify(arg, (_key, value) =>
      typeof value === 'bigint' ? value.toString() : value,
    );
  } catch {
    return String(arg);
  }
}

function splitLogArgs(args: unknown[]): { message: string; details: string | null } {
  if (args.length === 0) {
    return { message: '', details: null };
  }

  const [first, ...rest] = args;
  const messageParts = [first, ...rest.filter((arg) => typeof arg === 'string')];
  const detailParts = rest.filter((arg) => typeof arg !== 'string');

  return {
    message: messageParts.map(stringifyArg).join(' '),
    details: detailParts.length > 0 ? detailParts.map(stringifyArg).join('\n') : null,
  };
}

async function processQueue() {
  if (isProcessing || !activeDbHandler || pendingLogs.length === 0) return;
  isProcessing = true;

  while (pendingLogs.length > 0 && activeDbHandler) {
    const entry = pendingLogs.shift()!;
    try {
      await activeDbHandler(entry);
    } catch (err) {
      const originalError = (log as any)._original?.error;
      if (originalError) {
        originalError('Logger: failed to persist DB log', err);
      }
    }
  }

  isProcessing = false;
}

function pushToDb(level: string, tag: string, message: string, details?: string | null) {
  const entry: LogEntry = {
    level,
    tag,
    message,
    details: details ?? null,
    timestamp: new Date().toISOString(),
  };

  if (activeDbHandler) {
    pendingLogs.push(entry);
    void processQueue();
    return;
  }

  if (pendingLogs.length < 1000) {
    pendingLogs.push(entry);
  }
}

export function createLogger(tag: string): AppLogger {
  const persist = (level: string, args: unknown[]) => {
    const { message, details } = splitLogArgs(args);
    pushToDb(level, tag, message, details);
  };

  return {
    info: (...args) => persist('INFO', args),
    warn: (...args) => persist('WARN', args),
    error: (...args) => persist('ERROR', args),
    debug: (...args) => persist('DEBUG', args),
    critical: (...args) => {
      const logger = (log as any)._original?.error || console.error;
      logger(`[${new Date().toISOString()}]`, `[CRITICAL]`, `[${tag}]`, ...args);
      persist('CRITICAL', args);
    },
  };
}

export const log = createLogger('Global');

export function registerLogDbHandler(handler: DbHandler) {
  activeDbHandler = handler;
  void processQueue();
}

export function patchConsole(): void {
  if (typeof window === 'undefined') return;

  const originalLog = console.log;
  const originalWarn = console.warn;
  const originalError = console.error;
  const originalInfo = console.info;
  const originalDebug = console.debug;

  console.log = (...args: unknown[]) => log.info(...args);
  console.info = (...args: unknown[]) => log.info(...args);
  console.warn = (...args: unknown[]) => log.warn(...args);
  console.error = (...args: unknown[]) => log.error(...args);
  console.debug = (...args: unknown[]) => log.debug(...args);

  (log as any)._original = {
    log: originalLog,
    warn: originalWarn,
    error: originalError,
    info: originalInfo,
    debug: originalDebug,
  };
}
