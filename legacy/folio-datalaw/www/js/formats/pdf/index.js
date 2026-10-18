// PDF format descriptor for the registry.
import { mountPdf } from './mount.js';
import { pdfCover } from './cover.js';
export const pdf = { id: 'pdf', extensions: ['pdf'], mimes: ['application/pdf'], mount: mountPdf, cover: pdfCover };
