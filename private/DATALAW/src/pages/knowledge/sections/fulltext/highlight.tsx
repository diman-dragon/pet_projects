function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function buildHighlightedHtml(text: string, queries: string[]): string {
  const terms = queries
    .flatMap((query) =>
      query
        .toLowerCase()
        .split(/\s+/)
        .filter((term) => term.length > 1),
    )
    .filter(Boolean);

  if (!terms.length) {
    return escapeHtml(text);
  }

  const lowerText = text.toLowerCase();
  const marks: Array<[number, number]> = [];

  for (const term of terms) {
    let from = 0;
    while ((from = lowerText.indexOf(term, from)) !== -1) {
      marks.push([from, from + term.length]);
      from += 1;
    }
  }

  if (!marks.length) {
    return escapeHtml(text);
  }

  marks.sort((left, right) => left[0] - right[0]);

  const merged: Array<[number, number]> = [];
  for (const [start, end] of marks) {
    const last = merged[merged.length - 1];
    if (last && start <= last[1]) {
      last[1] = Math.max(last[1], end);
      continue;
    }
    merged.push([start, end]);
  }

  let cursor = 0;
  let html = '';
  for (const [start, end] of merged) {
    html += escapeHtml(text.slice(cursor, start));
    html += `<mark class="fts-mark">${escapeHtml(text.slice(start, end))}</mark>`;
    cursor = end;
  }

  return html + escapeHtml(text.slice(cursor));
}
