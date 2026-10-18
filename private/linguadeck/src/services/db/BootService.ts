import { exists } from "@tauri-apps/plugin-fs";
import { ConfigService } from "../ConfigService";
import { FileSystemService } from "../FileSystemService";
import { LibraryService } from "./LibraryService";

export const BootService = {
  async bootstrap(): Promise<string | null> {
    const savedPath = await ConfigService.getSavedLibraryPath();
    if (!savedPath || !(await exists(savedPath))) {
      return null;
    }

    await FileSystemService.initLibraryStructure(savedPath.replace(/[\\/]Linguadeck$/i, ""));
    await LibraryService.init(savedPath);
    return savedPath;
  }
};
