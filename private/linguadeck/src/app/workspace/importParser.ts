import type { CreateCardInput } from "../../services/db/CardService";
import { EMPTY_CARD } from "./defaults";

export function parseImport(raw: string): CreateCardInput[] {
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    // Пропускаем заголовок, если он есть (проверка по ключевому слову 'word' или 'front')
    .filter((line) => !line.toLowerCase().startsWith("word;") && !line.toLowerCase().startsWith("front;"))
    .map((line) => {
      const separator = line.includes("\t") ? "\t" : line.includes(";") ? ";" : ",";
      const parts = line.split(separator).map((part) => part.trim().replace(/^"|"$/g, ""));

      /**
       * Маппинг согласно структуре Python-скрипта (8 колонок):
       * 0: word (front)
       * 1: translation (back)
       * 2: transcription
       * 3: description (context)
       * 4: examples (example_1)
       * 5: audio_path
       * 6: video_path
       * 7: image_path (notes / или отдельное поле в модели)
       */
      return {
        ...EMPTY_CARD,
        front: parts[0] ?? "",
        back: parts[1] ?? "",
        transcription: parts[2] ?? "",
        context: parts[3] ?? "",
        example_1: parts[4] ?? "",
        // Поля медиафайлов
        audio_path: parts[5] ?? "",
        video_path: parts[6] ?? "",
        // Если в вашей модели notes используется для хранения пути к картинке или доп. заметок из поля 7
        notes: parts[7] ?? "",
        source: "import"
      };
    })
    .filter((item) => item.front && item.back);
}