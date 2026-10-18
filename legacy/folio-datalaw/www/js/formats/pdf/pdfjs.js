// Lazy loader for the vendored pdf.js (legacy build) and its worker; exposes the assets base URL.
export const ASSETS = new URL('../../../vendor/pdfjs/', import.meta.url).href;
let lib;
export function pdfjs() {
  lib ??= import('../../../vendor/pdfjs/pdf.min.mjs').then((m) => {
    m.GlobalWorkerOptions.workerSrc = ASSETS + 'pdf.worker.min.mjs';
    return m;
  });
  return lib;
}
