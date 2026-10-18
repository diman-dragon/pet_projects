// src/services/config/AppPaths.ts
//
// Singleton: хранит rootPath после инициализации.
// Заполняется в useSystemInit. Читается везде без пропсов.

let _rootPath: string | null = null;

export const AppPaths = {
  setRootPath(path: string) {
    _rootPath = path;
  },

  getRootPath(): string | null {
    return _rootPath;
  },

  // Возвращает путь к Knowledge_Base.
  // Используем нативный разделитель — на Windows это \, на Mac/Linux /
  getKbPath(): string | null {
    if (!_rootPath) return null;
    const sep = _rootPath.includes('\\') ? '\\' : '/';
    return _rootPath.replace(/[\\/]$/, '') + sep + 'Knowledge_Base';
  },

  getLegalCorpusPath(): string | null {
    if (!_rootPath) return null;
    const sep = _rootPath.includes('\\') ? '\\' : '/';
    return _rootPath.replace(/[\\/]$/, '') + sep + 'Legal_Corpus';
  },

  getPiiVaultPath(): string | null {
    if (!_rootPath) return null;
    const sep = _rootPath.includes('\\') ? '\\' : '/';
    return _rootPath.replace(/[\\/]$/, '') + sep + 'PII_Vault';
  },

  clear() {
    _rootPath = null;
  },
};
