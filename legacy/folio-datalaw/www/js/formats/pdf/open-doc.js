// Opens a PDF Blob as a pdf.js document (asks for a password when allowed).
import { pdfjs, ASSETS } from './pdfjs.js';
import { t } from '../../core/i18n.js';
export async function openPdf(blob, { askPassword = true } = {}) {
  const lib = await pdfjs();
  const data = new Uint8Array(await blob.arrayBuffer());
  const task = lib.getDocument({
    data,
    enableHWA: true, // GPU-backed canvases (the default software canvases are several times slower on phones)
    cMapUrl: ASSETS + 'cmaps/', cMapPacked: true,
    standardFontDataUrl: ASSETS + 'standard_fonts/',
    wasmUrl: ASSETS + 'wasm/', iccUrl: ASSETS + 'iccs/',
  });
  task.onPassword = (update) => {
    const pw = askPassword ? prompt(t('viewer.password')) : null;
    if (pw === null) task.destroy(); else update(pw);
  };
  return task.promise;
}
