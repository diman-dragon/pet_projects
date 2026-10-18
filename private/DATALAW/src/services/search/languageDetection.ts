// src/services/search/languageDetection.ts
//
// Лёгкая детекция языка текста без внешних зависимостей.
// Используется при индексации (DocumentIndexer, indexSingleFile) и
// в QueryProcessor (итерация 3) для определения языка запроса.
//
// Поддерживаемые языки: ru, en, de, fr, uk (украинский).
// При низкой уверенности (confidence < MIN_CONFIDENCE) возвращает null.
// Fallback: язык приложения (задаётся через detectLanguage второй аргумент).

export interface DetectionResult {
  lang: string;       // ISO-639-1
  confidence: number; // 0..1
}

const MIN_CONFIDENCE = 0.35;

// Характерные n-граммы и слова для каждого языка.
// Ключи — всё в нижнем регистре без диакритики.
const LANG_PATTERNS: Record<string, RegExp[]> = {
  ru: [
    /[а-яё]{3,}/g,                  // кириллица
    /\b(и|в|не|на|что|как|это|для|по|из|от|же|но|или|так|уже|при|там|где|когда|если|чтобы|только|также|его|её|их|был|была|было|будет|есть|нет|все|тот|эта|это|эти)\b/g,
  ],
  uk: [
    /[іїєґ]/g,                        // специфические украинские буквы
    /\b(і|в|не|на|що|як|це|для|по|із|від|же|але|або|так|вже|при|там|де|коли|якщо|щоб|тільки|також|його|її|їх|був|була|було|буде|є|ні|всі|той|ця|це|ці)\b/g,
  ],
  en: [
    /\b[a-z]{3,}\b/g,
    /\b(the|and|or|in|on|at|to|of|for|is|are|was|were|be|have|has|had|with|that|this|from|by|not|but|as|an|it|its|if|so|do|did|will|can|would|could|should|may|might|shall)\b/g,
  ],
  de: [
    /[äöüß]/g,
    /\b(der|die|das|und|in|von|mit|ist|auf|für|nicht|ein|eine|dem|den|des|zu|bei|auch|nach|es|er|sie|wir|ich|sie|haben|sein|werden|kann|wird|war|wurden|einem|einer)\b/g,
  ],
  fr: [
    /[àâéèêëîïôùûüç]/g,
    /\b(le|la|les|de|du|des|un|une|et|en|dans|pour|par|sur|avec|est|sont|pas|que|qui|je|il|elle|nous|vous|ils|elles|être|avoir|faire|au|aux)\b/g,
  ],
};

/**
 * Подсчитывает суммарное количество совпадений всех паттернов языка.
 */
function scoreText(text: string, patterns: RegExp[]): number {
  return patterns.reduce((sum, re) => {
    const matches = text.match(re);
    return sum + (matches ? matches.length : 0);
  }, 0);
}

/**
 * Определяет язык текста.
 *
 * @param text        — анализируемый текст (первые ~2000 символов для скорости)
 * @param fallbackLang — язык по умолчанию при низкой уверенности (null = вернуть null)
 */
export function detectLanguage(
  text: string,
  fallbackLang: string | null = null,
): DetectionResult | null {
  const sample = text.slice(0, 2000).toLowerCase();
  if (sample.trim().length < 20) return null;

  const scores: Record<string, number> = {};
  for (const [lang, patterns] of Object.entries(LANG_PATTERNS)) {
    scores[lang] = scoreText(sample, patterns);
  }

  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const top = entries[0];
  const total = entries.reduce((s, e) => s + e[1], 0);

  if (total === 0) {
    if (fallbackLang) return { lang: fallbackLang, confidence: 0 };
    return null;
  }

  const confidence = top[1] / total;

  if (confidence < MIN_CONFIDENCE) {
    if (fallbackLang) return { lang: fallbackLang, confidence };
    return null;
  }

  return { lang: top[0], confidence };
}

/**
 * Удобная обёртка: возвращает только ISO-код или null.
 */
export function detectLang(text: string, fallbackLang: string | null = null): string | null {
  return detectLanguage(text, fallbackLang)?.lang ?? null;
}
