import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { FileSystemService } from "../services/FileSystemService";
import { LibraryService } from "../services/db/LibraryService";
import { ConfigService } from "../services/ConfigService";

export function useAppInitialization(onReady: (path: string) => void) {
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function selectFolder() {
    const value = await open({
      directory: true,
      multiple: false,
      title: "Выберите папку для библиотеки Linguadeck",
    });
    if (typeof value === "string") {
      setSelectedPath(value);
      setError(null);
    }
  }

  async function initialize() {
    if (!selectedPath) return;
    setBusy(true);
    setError(null);
    try {
      const finalPath = await FileSystemService.initLibraryStructure(selectedPath);
      await LibraryService.init(finalPath);
      await ConfigService.saveLibraryPath(finalPath);
      onReady(finalPath);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось подготовить библиотеку.");
    } finally {
      setBusy(false);
    }
  }

  return {
    selectedPath,
    busy,
    error,
    selectFolder,
    initialize,
  };
}
