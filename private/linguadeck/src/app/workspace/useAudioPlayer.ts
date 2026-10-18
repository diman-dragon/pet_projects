import { useCallback, useRef, useState } from "react";
import { AudioService } from "../../services/AudioService";

interface UseAudioPlayerOptions {
  /** Root of the Linguadeck library: D:\Projekts\Linguadeck */
  libraryPath: string;
}

export function useAudioPlayer({ libraryPath }: UseAudioPlayerOptions) {
  const [playing, setPlaying] = useState<string | null>(null);
  const libraryPathRef = useRef(libraryPath);
  libraryPathRef.current = libraryPath;

  const playAudio = useCallback(async (audioPath: string): Promise<void> => {
    setPlaying(audioPath);
    try {
      await AudioService.play(audioPath, libraryPathRef.current);
    } finally {
      setPlaying(null);
    }
  }, []);

  const stopAudio = useCallback(() => {
    AudioService.stop();
    setPlaying(null);
  }, []);

  return { playAudio, stopAudio, playing };
}
