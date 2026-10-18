import { convertFileSrc } from "@tauri-apps/api/core";
import { join } from "@tauri-apps/api/path";
import { exists } from "@tauri-apps/plugin-fs";

let currentAudio: HTMLAudioElement | null = null;

/**
 * Resolve card's audio_path to an absolute path.
 *
 * libraryPath = D:\Projekts\Linguadeck  (the Linguadeck root folder)
 *
 * audio_path stored in card:
 *   "media/audio/Oxford_3000/value.mp3"
 *   → D:\Projekts\Linguadeck\media\audio\Oxford_3000\value.mp3
 *
 *   "audio/file.mp3"
 *   → D:\Projekts\Linguadeck\media\audio\file.mp3  (joined via media/)
 *
 *   Absolute path → used as-is
 */
async function resolveAudioPath(audioPath: string, libraryPath: string): Promise<string> {
  const trimmed = audioPath.trim();

  // Already absolute (Windows: C:\ or D:\, Unix: /)
  if (/^([A-Za-z]:[/\\]|\/|\\\\)/.test(trimmed)) {
    return trimmed;
  }

  // Relative — join with libraryPath directly.
  // card stores "media/audio/Oxford_3000/value.mp3"
  // libraryPath is "D:\Projekts\Linguadeck"
  // result: "D:\Projekts\Linguadeck\media\audio\Oxford_3000\value.mp3"  ✓
  return join(libraryPath, trimmed);
}

export const AudioService = {
  async play(audioPath: string, libraryPath: string): Promise<void> {
    if (!audioPath?.trim()) return;

    if (currentAudio) {
      currentAudio.pause();
      currentAudio.src = "";
      currentAudio = null;
    }

    const filePath = await resolveAudioPath(audioPath, libraryPath);
    const fileExists = await exists(filePath).catch(() => false);

    if (!fileExists) {
      throw new Error(`Аудиофайл не найден: ${filePath}`);
    }

    const assetUrl = convertFileSrc(filePath);
    const audio = new Audio(assetUrl);
    currentAudio = audio;

    return new Promise((resolve, reject) => {
      audio.onended = () => { currentAudio = null; resolve(); };
      audio.onerror = () => {
        currentAudio = null;
        reject(new Error(`Не удалось воспроизвести: ${filePath}`));
      };
      audio.play().catch(reject);
    });
  },

  stop() {
    if (currentAudio) {
      currentAudio.pause();
      currentAudio.src = "";
      currentAudio = null;
    }
  },
};
