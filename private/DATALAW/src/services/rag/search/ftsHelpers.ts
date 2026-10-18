import { stemToken } from '../utils/stemming';

export function tokenizeQuery(query: string): string[] {
  return Array.from(
    new Set(
      query
        .replace(/[^\p{L}\p{N}\s]/gu, ' ')
        .trim()
        .split(/\s+/)
        .map((token) => token.trim())
        .filter((token) => token.length > 1)
        .slice(0, 8),
    ),
  );
}

export function buildFtsExpression(tokens: string[], expandedTerms: string[] = []): string {
  const terms = new Set<string>();

  for (const token of tokens) {
    terms.add(`"${token.replace(/"/g, '""')}\"*`);
    const stem = stemToken(token);
    if (stem && stem !== token) {
      terms.add(`"${stem.replace(/"/g, '""')}\"*`);
    }
  }

  for (const term of expandedTerms) {
    const lower = term.toLowerCase();
    if (lower.length > 2) {
      terms.add(`"${lower.replace(/"/g, '""')}\"*`);
    }
  }

  return Array.from(terms).join(' OR ');
}

export function buildSearchTerms(query: string, expandedTerms: string[] = []): { tokens: string[]; searchTerms: string[] } {
  const tokens = tokenizeQuery(query).map((token) => token.toLocaleLowerCase());
  const searchTerms: string[] = [];

  for (const token of tokens) {
    searchTerms.push(token);
    const stem = stemToken(token);
    if (stem && stem !== token) {
      searchTerms.push(stem);
    }
  }

  for (const term of expandedTerms) {
    const lower = term.toLowerCase();
    if (lower.length > 2) {
      searchTerms.push(lower);
    }
  }

  return { tokens, searchTerms };
}

export function calculateScanScore(
  title: string,
  chunkText: string,
  tokens: string[],
  searchTerms: string[],
): number {
  const titleLower = title.toLocaleLowerCase();
  const haystack = `${title}\n${chunkText}`.toLocaleLowerCase();
  let exactHits = 0;
  let partialHits = 0;

  for (const term of searchTerms) {
    if (haystack.includes(term)) {
      exactHits += 1;
    } else {
      const prefix = term.slice(0, Math.max(4, term.length - 2));
      if (prefix.length >= 4 && haystack.includes(prefix)) {
        partialHits += 1;
      }
    }
  }

  const titleBoost = searchTerms.some((term) => titleLower.includes(term)) ? 0.15 : 0;
  return Math.min(
    (exactHits / tokens.length) * 0.8 +
    (partialHits / tokens.length) * 0.35 +
    (exactHits > 0 ? 0.05 : 0) +
    titleBoost,
    0.95,
  );
}
