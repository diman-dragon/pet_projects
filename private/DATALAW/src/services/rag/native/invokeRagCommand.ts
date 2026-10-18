import { invoke } from '@tauri-apps/api/core';

export function invokeRagCommand<T>(
  command: string,
  payload?: Record<string, unknown>,
): Promise<T> {
  return invoke<T>(command, payload);
}
