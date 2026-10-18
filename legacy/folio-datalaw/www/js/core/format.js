// Human-readable file size and date.
import { resolve } from './i18n.js';
export function fmtSize(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}
export const fmtDate = (ts) => new Date(ts).toLocaleDateString(resolve());
