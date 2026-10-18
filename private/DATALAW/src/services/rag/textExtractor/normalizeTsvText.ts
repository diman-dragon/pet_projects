// src/services/rag/textExtractor/normalizeTsvText.ts
//
// Единая утилита нормализации TSV/CSV текста — используется при:
//   • индексации (plain.ts)
//   • отображении чанков в UI (ChunkDetail, VectorResultCard)
//   • передаче контекста агентам (KnowledgeSearchAgent, LegalResearchAgent)
//
// Вход: строки вида  «Битва на Косовом поле;15 июня 1389;Описание...;;;;»
// Выход: «Битва на Косовом поле (15 июня 1389): Описание...»
//
// ВАЖНО: строки разделяются через \n\n — это сигнал чанкеру что каждая
// запись является самостоятельным параграфом и НЕ должна склеиваться
// с соседними записями в один чанк.

const PATH_HEADERS_RE = /^(audio|video|image|file|src|href|url|.*_path|.*_url)$/i;

function rowToSentence(headers: string[], cols: string[]): string {
  const pairs: [string, string][] = [];
  for (let i = 0; i < headers.length; i++) {
    const h = headers[i].trim();
    const v = (cols[i] ?? '').trim();
    if (!v || PATH_HEADERS_RE.test(h)) continue;
    pairs.push([h, v]);
  }
  if (!pairs.length) return '';

  const [first, second, ...rest] = pairs;
  let sentence = first[1];
  if (second) sentence += ` (${second[1]})`;
  const tail = rest.map(([, v]) => v).filter(Boolean).join('. ');
  if (tail) sentence += ': ' + tail;
  return sentence;
}

/**
 * Нормализует текст, который может быть сырым TSV/CSV.
 * Если текст не является TSV — возвращает как есть.
 *
 * Строки разделяются через \n\n (двойной перенос), чтобы
 * Rust-чанкер выбирал separator="\n\n" и не склеивал записи.
 */
export function normalizeTsvText(raw: string): string {
  if (!raw) return raw;
  const lines = raw.split('\n').map((l) => l.trim()).filter(Boolean);
  if (lines.length < 1) return raw;

  // Определяем разделитель
  const sample = lines.find((l) => l.includes(';') || l.includes('\t')) ?? lines[0];
  let d: string | null = null;
  if (sample.split(';').length >= 3) d = ';';
  else if (sample.split('\t').length >= 3) d = '\t';
  if (!d) return raw;

  // Заголовочная строка — только буквы/цифры/подчёркивание/пробелы
  const firstCols = lines[0].split(d).map((h) => h.trim());
  const looksLikeHeader =
    firstCols.length >= 2 && firstCols.every((h) => /^[a-zA-Zа-яёА-ЯЁ_\s]+$/.test(h));

  const headers = looksLikeHeader ? firstCols : null;
  const dataStart = looksLikeHeader ? 1 : 0;

  const result: string[] = [];
  for (let i = dataStart; i < lines.length; i++) {
    const cols = lines[i].split(d);
    if (headers) {
      const sentence = rowToSentence(headers, cols);
      if (sentence) result.push(sentence);
    } else {
      // Нет заголовка — убираем пустые колонки и path-значения
      const clean = cols
        .map((v) => v.trim())
        .filter((v) => v && !v.match(/\.(mp3|mp4|wav|jpg|jpeg|png|gif|webp)$/i))
        .join(' ');
      if (clean) result.push(clean);
    }
  }

  // \n\n между записями — чанкер воспринимает это как границы параграфов
  // и НЕ будет склеивать разные записи в один чанк
  return result.length ? result.join('\n\n') : raw;
}

/**
 * Быстрая проверка — является ли текст сырым TSV/CSV.
 */
export function isTsvText(text: string): boolean {
  const first = (text.split('\n')[0] ?? '').trim();
  return first.split(';').length >= 3 || first.split('\t').length >= 3;
}

/**
 * Разбивает нормализованный TSV на отдельные записи (параграфы).
 * Используется в indexSingleFile для создания чанка на каждую запись.
 */
export function splitTsvRecords(normalizedText: string): string[] {
  return normalizedText
    .split('\n\n')
    .map((s) => s.trim())
    .filter(Boolean);
}
