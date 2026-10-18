import { collectFiles } from './collectFiles';
import { indexSingleFile } from './indexSingleFile';
import type { IndexFolderOptions } from './types';

export async function indexFolder(
  folderPath: string,
  opts?: IndexFolderOptions,
): Promise<{ indexed: number; skipped: number; errors: string[] }> {
  const files = await collectFiles(folderPath, opts?.signal);
  const errors: string[] = [];
  let indexed = 0;
  let skipped = 0;

  for (let index = 0; index < files.length; index += 1) {
    if (opts?.signal?.aborted) break;

    const file = files[index];
    opts?.onFileProgress?.(file.name, index, files.length);

    try {
      const result = await indexSingleFile({
        filePath: file.fullPath,
        category: file.category,
        signal: opts?.signal,
        skipEmbeddings: opts?.skipEmbeddings,
        onProgress: opts?.onChunkProgress,
      });

      if (result.skipped) skipped += 1;
      else indexed += 1;
    } catch (error: any) {
      errors.push(`${file.name}: ${error?.message ?? String(error)}`);
    }
  }

  return { indexed, skipped, errors };
}
