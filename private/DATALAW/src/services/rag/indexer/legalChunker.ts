// src/services/rag/indexer/legalChunker.ts
//
// Структурный чанкер для юридических документов.
//
// СТРАТЕГИЯ:
// 1. Обнаруживаем тип документа (НПА, договор, процессуальный, решение суда)
// 2. Разбиваем по иерархии: Часть → Раздел → Глава → Статья → Пункт
// 3. В каждый чанк добавляем заголовок-хлебные крошки:
//    "[ГК РФ | Глава 29 | Статья 450] текст..."
//    Это критически улучшает качество эмбеддингов — модель видит контекст нормы.
// 4. Если статья слишком длинная (> maxChunkChars) — разбиваем по пунктам.
// 5. Если пункт тоже длинный — по предложениям с перекрытием.
//
// ФОРМАТЫ:
//   НПА/Кодекс:   Статья 1., Ст. 1, Статья 1.1.
//   Договор:      1. ПРЕДМЕТ, 1.1. Исполнитель..., 1.1.1.
//   Процессуальный: «РЕШИЛ:», «УСТАНОВИЛ:», абзацы
//   Решение суда: нумерованные абзацы иска/мотивировки/резолюции

import type { TextChunk } from '../ChunkService';

// ── Конфигурация ─────────────────────────────────────────────────────────────

const MAX_CHUNK_CHARS = 900;     // максимальный размер чанка (≈180 токенов)
const MIN_CHUNK_CHARS = 80;      // минимальный — меньше не индексируем
const OVERLAP_CHARS = 120;       // перекрытие между пунктами внутри статьи
const MAX_HEADER_LENGTH = 120;   // обрезаем длинные заголовки в breadcrumb

// ── Определение типа документа ──────────────────────────────────────────────

export type LegalDocType =
  | 'codex'        // кодекс / НПА (Статья N.)
  | 'contract'     // договор (1.1. Исполнитель...)
  | 'procedural'   // процессуальный документ (УСТАНОВИЛ:, РЕШИЛ:)
  | 'court'        // судебное решение / определение
  | 'generic';     // всё остальное

export function detectLegalDocType(text: string): LegalDocType {
  const sample = text.slice(0, 3000);

  // Судебные акты
  if (/РЕШИЛ[:\s]|ОПРЕДЕЛИЛ[:\s]|ПОСТАНОВИЛ[:\s]|р\s*е\s*ш\s*и\s*л/i.test(sample)) return 'court';
  if (/УСТАНОВИЛ[:\s]|МОТИВИРОВОЧНАЯ ЧАСТЬ|дело №\s*\d/i.test(sample)) return 'procedural';

  // НПА — ищем паттерн «Статья N.» или § N
  const articleMatches = (text.match(/\bСтатья\s+\d+/gi) ?? []).length;
  if (articleMatches >= 2) return 'codex';

  // Договор — нумерованные разделы «1.» «1.1.» «2.»
  // FIX: порог снижен до 2, regex исправлен (пробелов может быть любое количество)
  const contractMatches = (text.match(/^\d{1,2}\.\s+[А-ЯЁIVXLCDM«"]/gm) ?? []).length;
  if (contractMatches >= 2) return 'contract';

  return 'generic';
}

export function isLegalDocument(text: string): boolean {
  return detectLegalDocType(text) !== 'generic';
}

// ── Утилиты ─────────────────────────────────────────────────────────────────

function trimHeader(s: string): string {
  const t = s.replace(/\s+/g, ' ').trim();
  return t.length > MAX_HEADER_LENGTH ? t.slice(0, MAX_HEADER_LENGTH - 1) + '…' : t;
}

function buildBreadcrumb(parts: (string | null)[]): string {
  const valid = parts.filter(Boolean) as string[];
  return valid.length ? '[' + valid.join(' | ') + '] ' : '';
}

function splitBySentences(text: string, maxChars: number, overlap: number): string[] {
  // Разбиваем по знакам препинания сохраняя юридические сокращения
  const sentenceRe = /(?<=[.!?»])\s+(?=[А-ЯA-Z0-9«"])/g;
  const sentences = text.split(sentenceRe).filter(s => s.trim().length > 0);

  const chunks: string[] = [];
  let current = '';
  for (const sent of sentences) {
    if (current.length + sent.length + 1 > maxChars && current.length > 0) {
      chunks.push(current.trim());
      // Берём последнее предложение как начало следующего чанка (overlap)
      const words = current.slice(-overlap).trim();
      current = words + ' ' + sent;
    } else {
      current = current ? current + ' ' + sent : sent;
    }
  }
  if (current.trim().length >= MIN_CHUNK_CHARS) chunks.push(current.trim());
  return chunks;
}

function splitLongText(text: string, maxChars: number): string[] {
  if (text.length <= maxChars) return [text];

  // Попытка 1: по пунктам (нумерованные абзацы)
  const paragraphRe = /(?=^\s*(?:\d+\.\s|\p{L}\)\s|[—–]\s))/mu;
  const parts = text.split(paragraphRe).filter(p => p.trim().length >= MIN_CHUNK_CHARS);
  if (parts.length > 1) {
    const result: string[] = [];
    for (const p of parts) {
      if (p.length > maxChars) result.push(...splitBySentences(p, maxChars, OVERLAP_CHARS));
      else result.push(p.trim());
    }
    return result;
  }

  // Попытка 2: по двойному переводу строки (абзацы)
  const paraBreak = text.split(/\n{2,}/).filter(p => p.trim().length >= MIN_CHUNK_CHARS);
  if (paraBreak.length > 1) {
    const result: string[] = [];
    let current = '';
    for (const para of paraBreak) {
      if (current.length + para.length + 2 > maxChars && current.length > 0) {
        result.push(current.trim());
        current = para.trim();
      } else {
        current = current ? current + '\n\n' + para.trim() : para.trim();
      }
    }
    if (current.trim().length >= MIN_CHUNK_CHARS) result.push(current.trim());
    return result;
  }

  // Попытка 3: по предложениям
  return splitBySentences(text, maxChars, OVERLAP_CHARS);
}

// ── Чанкер для НПА / Кодексов ───────────────────────────────────────────────

interface LegalSection {
  type: 'part' | 'section' | 'chapter' | 'article' | 'paragraph';
  header: string;  // «Статья 450. Изменение и расторжение договора»
  body: string;    // тело без заголовка
}

// Паттерны заголовков НПА (порядок важен — от крупного к мелкому)
// @ts-ignore – reserved for future NPA chunking
const CODEX_SPLITTERS: Array<{ re: RegExp; type: LegalSection['type'] }> = [
  { re: /^(?:ЧАСТЬ\s+[А-ЯIVXLCDMа-яёivxlcdm]+|Часть\s+(?:первая|вторая|третья|четвёртая|пятая|\d+))/m, type: 'part' },
  { re: /^(?:РАЗДЕЛ\s+[А-ЯIVXLCDM]+\.?|Раздел\s+[А-ЯIVXLCDMа-я0-9]+\.?)/m, type: 'section' },
  { re: /^(?:ГЛАВА\s+\d+(?:\.\d+)?\.?|Глава\s+\d+(?:\.\d+)?\.?)/m, type: 'chapter' },
  { re: /^(?:Статья\s+\d+(?:\.\d+)?\.?|СТ\.\s*\d+(?:\.\d+)?|§\s*\d+)/m, type: 'article' },
];

function splitCodexByStructure(text: string): LegalSection[] {
  // Строим единое регулярное выражение для всех уровней
  const combinedRe = new RegExp(
    '(' +
    [
      /(?:ЧАСТЬ|Часть)\s+(?:[А-ЯIVXLCDMа-яёivxlcdm]+|первая|вторая|третья|четвёртая|пятая|\d+)/,
      /(?:РАЗДЕЛ|Раздел)\s+[А-ЯIVXLCDMа-яё0-9]+\.?/,
      /(?:ГЛАВА|Глава)\s+\d+(?:\.\d+)?\.?/,
      /Статья\s+\d+(?:\.\d+)?\.?\s*[А-Яа-яёA-Za-z(]?[^\n]*/,
      /§\s*\d+(?:\.\d+)?\s*[^\n]*/,
    ].map(r => r.source).join('|') +
    ')',
    'gm',
  );

  const sections: LegalSection[] = [];
  let lastIndex = 0;
  let lastHeader = '';
  let lastType: LegalSection['type'] = 'paragraph';

  const matches = [...text.matchAll(combinedRe)];
  for (const match of matches) {
    const matchStart = match.index ?? 0;
    if (matchStart > lastIndex) {
      const body = text.slice(lastIndex, matchStart).trim();
      if (body.length >= MIN_CHUNK_CHARS) {
        sections.push({ type: lastType, header: lastHeader, body });
      }
    }
    lastHeader = match[0].trim();
    lastType = classifyHeader(lastHeader);
    lastIndex = matchStart + match[0].length;
  }

  // Последний блок
  const tail = text.slice(lastIndex).trim();
  if (tail.length >= MIN_CHUNK_CHARS) {
    sections.push({ type: lastType, header: lastHeader, body: tail });
  }

  return sections;
}

function classifyHeader(header: string): LegalSection['type'] {
  const h = header.toUpperCase();
  if (/^(ЧАСТЬ|ЧАСТЬ\s+ПЕРВАЯ|ЧАСТЬ\s+ВТОРАЯ)/.test(h)) return 'part';
  if (/^РАЗДЕЛ/.test(h)) return 'section';
  if (/^ГЛАВА/.test(h)) return 'chapter';
  if (/^(СТАТЬЯ|СТ\.|§)/.test(h)) return 'article';
  return 'paragraph';
}

function chunkCodex(text: string, docTitle: string): TextChunk[] {
  const sections = splitCodexByStructure(text);
  const chunks: TextChunk[] = [];
  let charPos = 0;

  // Контекстный стек — запоминаем заголовки родительских уровней
  const context: Record<LegalSection['type'], string> = {
    part: '', section: '', chapter: '', article: '', paragraph: '',
  };

  const hierarchy: Array<LegalSection['type']> = ['part', 'section', 'chapter', 'article', 'paragraph'];

  for (const section of sections) {
    // Обновляем контекст — сбрасываем дочерние уровни
    const level = hierarchy.indexOf(section.type);
    context[section.type] = trimHeader(section.header);
    for (let i = level + 1; i < hierarchy.length; i++) {
      context[hierarchy[i]] = '';
    }

    // Строим хлебные крошки
    const breadcrumb = buildBreadcrumb([
      docTitle ? trimHeader(docTitle) : null,
      context.part || null,
      context.section || null,
      context.chapter || null,
      section.type === 'article' ? context.article : null,
    ]);

    const fullText = breadcrumb + section.header + '\n' + section.body;

    // Если чанк в пределах лимита — берём целиком
    if (fullText.length <= MAX_CHUNK_CHARS) {
      if (fullText.trim().length >= MIN_CHUNK_CHARS) {
        chunks.push({
          text: fullText,
          chunkIndex: chunks.length,
          charStart: charPos,
          charEnd: charPos + fullText.length,
        });
        charPos += fullText.length + 2;
      }
    } else {
      // Разбиваем тело на части, сохраняя breadcrumb в каждой
      const bodyParts = splitLongText(section.body, MAX_CHUNK_CHARS - breadcrumb.length - section.header.length - 1);
      for (const part of bodyParts) {
        if (part.trim().length < MIN_CHUNK_CHARS) continue;
        const chunkText = breadcrumb + section.header + '\n' + part;
        chunks.push({
          text: chunkText,
          chunkIndex: chunks.length,
          charStart: charPos,
          charEnd: charPos + chunkText.length,
        });
        charPos += chunkText.length + 2;
      }
    }
  }

  return chunks;
}

// ── Чанкер для договоров ─────────────────────────────────────────────────────

// @ts-ignore – reserved for future contract chunking
const CONTRACT_SECTION_RE = /^(\s*)(\d{1,2})\.(\d{0,2})\.?(\d{0,2})\.?\s+/m;

interface ContractSection {
  number: string;  // «1.2.3»
  depth: number;   // 1=раздел, 2=пункт, 3=подпункт
  header: string;
  body: string;
}

function splitContractSections(text: string): ContractSection[] {
  const sectionRe = /^(\d{1,2})(?:\.(\d{1,3}))?(?:\.(\d{1,3}))?\.?\s+([^\n]+)/gm;
  const sections: ContractSection[] = [];
  let lastIndex = 0;
  let lastSection: ContractSection | null = null;

  const matches = [...text.matchAll(sectionRe)];
  for (const match of matches) {
    const matchStart = match.index ?? 0;

    if (lastSection && matchStart > lastIndex) {
      lastSection.body = text.slice(lastIndex, matchStart).trim();
      sections.push(lastSection);
    }

    const n1 = match[1];
    const n2 = match[2];
    const n3 = match[3];
    const number = [n1, n2, n3].filter(Boolean).join('.');
    const depth = [n1, n2, n3].filter(Boolean).length;

    lastSection = {
      number,
      depth,
      header: match[0].trim(),
      body: '',
    };
    lastIndex = matchStart + match[0].length;
  }

  if (lastSection) {
    lastSection.body = text.slice(lastIndex).trim();
    sections.push(lastSection);
  }

  return sections;
}

function chunkContract(text: string, docTitle: string): TextChunk[] {
  const sections = splitContractSections(text);
  if (sections.length < 2) return []; // не похоже на договор — fallback

  const chunks: TextChunk[] = [];
  let charPos = 0;

  // Группируем: пункты верхнего уровня объединяем с их подпунктами
  // до достижения MAX_CHUNK_CHARS
  let current = '';
  let _currentHeader = '';
  void _currentHeader; // объявлен для будущего breadcrumb
  let currentStart = charPos;

  for (const section of sections) {
    const titleBreadcrumb = docTitle ? `[${trimHeader(docTitle)} | ${section.number}] ` : `[${section.number}] `;
    const sectionText = section.depth === 1
      ? `${titleBreadcrumb}${section.header}\n${section.body}`
      : `  ${section.number}. ${section.body}`;

    if (section.depth === 1 && current.length > 0) {
      // Новый раздел — сохраняем предыдущий
      if (current.length >= MIN_CHUNK_CHARS) {
        chunks.push({ text: current.trim(), chunkIndex: chunks.length, charStart: currentStart, charEnd: currentStart + current.length });
      }
      currentStart = charPos;
      current = '';
    }

    if (current.length + sectionText.length > MAX_CHUNK_CHARS && current.length > 0) {
      // Текущий чанк переполнен — закрываем его
      if (current.length >= MIN_CHUNK_CHARS) {
        chunks.push({ text: current.trim(), chunkIndex: chunks.length, charStart: currentStart, charEnd: currentStart + current.length });
        charPos += current.length + 2;
      }
      currentStart = charPos;
      current = sectionText;
    } else {
      current = current ? current + '\n' + sectionText : sectionText;
    }
  }

  if (current.trim().length >= MIN_CHUNK_CHARS) {
    chunks.push({ text: current.trim(), chunkIndex: chunks.length, charStart: currentStart, charEnd: currentStart + current.length });
  }

  return chunks;
}

// ── Чанкер для судебных актов / процессуальных документов ───────────────────

const COURT_STRUCTURAL_RE = /^(УСТАНОВИЛ[:\s]|РЕШИЛ[:\s]|ОПРЕДЕЛИЛ[:\s]|ПОСТАНОВИЛ[:\s]|МОТИВИРОВОЧНАЯ ЧАСТЬ|РЕЗОЛЮТИВНАЯ ЧАСТЬ|ОПИСАТЕЛЬНАЯ ЧАСТЬ)/m;

function chunkCourtDocument(text: string, docTitle: string): TextChunk[] {
  const chunks: TextChunk[] = [];
  let charPos = 0;

  // Разбиваем по ключевым блокам судебного акта
  const parts = text.split(/(УСТАНОВИЛ[:\s]|РЕШИЛ[:\s]|ОПРЕДЕЛИЛ[:\s]|ПОСТАНОВИЛ[:\s]|МОТИВИРОВОЧНАЯ ЧАСТЬ|РЕЗОЛЮТИВНАЯ ЧАСТЬ|ОПИСАТЕЛЬНАЯ ЧАСТЬ)/i);

  let partLabel = '';
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i].trim();
    if (!part) continue;

    if (COURT_STRUCTURAL_RE.test(part + ':')) {
      partLabel = part.trim();
      continue;
    }

    const breadcrumb = buildBreadcrumb([
      docTitle ? trimHeader(docTitle) : null,
      partLabel || null,
    ]);

    const subParts = splitLongText(part, MAX_CHUNK_CHARS - breadcrumb.length);
    for (const sub of subParts) {
      if (sub.trim().length < MIN_CHUNK_CHARS) continue;
      const chunkText = breadcrumb + sub;
      chunks.push({
        text: chunkText,
        chunkIndex: chunks.length,
        charStart: charPos,
        charEnd: charPos + chunkText.length,
      });
      charPos += chunkText.length + 2;
    }
  }

  return chunks;
}

// ── Главная точка входа ──────────────────────────────────────────────────────

/**
 * Основной юридический чанкер.
 * Возвращает [] если не удалось определить структуру — тогда indexSingleFile
 * падает на стандартный ChunkService.chunk().
 */
export function chunkLegalDocument(text: string, docTitle: string): TextChunk[] {
  const docType = detectLegalDocType(text);

  let chunks: TextChunk[];

  switch (docType) {
    case 'codex':
      chunks = chunkCodex(text, docTitle);
      break;
    case 'contract':
      chunks = chunkContract(text, docTitle);
      if (chunks.length < 2) {
        // Договор не распознан — возвращаем пусто, вызывающий код упадёт на generic
        return [];
      }
      break;
    case 'court':
    case 'procedural':
      chunks = chunkCourtDocument(text, docTitle);
      break;
    default:
      return [];
  }

  if (chunks.length === 0) return [];

  // Финальная нормализация chunkIndex (после всех манипуляций может нарушиться)
  return chunks.map((c, i) => ({ ...c, chunkIndex: i }));
}
