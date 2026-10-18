class MasterKeyServiceClass {
  setStatusDbPath(_dbPath: string): void {}

  async unlockWithPassword(_password: string): Promise<string> {
    return 'DUMMY_KEY';
  }

  async ensureSessionKey(): Promise<string> {
    return 'DUMMY_KEY';
  }

  getSqlCipherKey(): string | null {
    return null;
  }

  isUnlocked(): boolean {
    return true;
  }

  async getAesKey(): Promise<CryptoKey | null> {
    return null;
  }

  async getEncryptionSeed(): Promise<Uint8Array | null> {
    return null;
  }

  lock(): void {}
}

export const MasterKeyService = new MasterKeyServiceClass();
