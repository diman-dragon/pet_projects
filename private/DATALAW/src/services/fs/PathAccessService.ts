import { AppPaths } from '../config/AppPaths';
import { isPathInside, normalizePath } from './pathAccessCore';

export const PathAccessService = {
  isWorkspacePath(path: string): boolean {
    const rootPath = AppPaths.getRootPath();
    const normalizedPath = normalizePath(path);

    // Allow access to all files if rootPath is not yet set (e.g. during initial dialog)
    // or if the path is actually inside the workspace.
    if (!rootPath) return true;

    // Check if it's inside workspace
    if (isPathInside(rootPath, normalizedPath)) return true;

    // Also allow common temp/downloads for upload purposes via normalized segments.
    const segments = normalizedPath.split('\\').filter(Boolean);
    if (segments.includes('downloads') || segments.includes('temp') || segments.includes('desktop')) {
      return true;
    }

    return false;
  },

  assertWorkspacePath(path: string): void {
    if (!this.isWorkspacePath(path)) {
      throw new Error(`Path is outside the Dorlane workspace: ${path}`);
    }
  },

  assertWorkspaceMove(oldPath: string, newPath: string): void {
    this.assertWorkspacePath(oldPath);
    this.assertWorkspacePath(newPath);
  },
};
