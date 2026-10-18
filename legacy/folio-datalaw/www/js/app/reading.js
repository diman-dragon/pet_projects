// "Reading now" rule: documents that were started (past page 1) and are not finished, most recently opened first.
export const isReading = (d) => (d.lastPage ?? 1) > 1 && !(d.pages > 0 && d.lastPage >= d.pages);
export const readingDocs = (docs, max = 12) => docs.filter(isReading).slice(0, max);
