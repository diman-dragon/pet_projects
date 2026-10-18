import { StorageService } from './StorageService';

export async function secureDelete(filePath: string): Promise<void> {
  await StorageService.secureDelete(filePath);
}
