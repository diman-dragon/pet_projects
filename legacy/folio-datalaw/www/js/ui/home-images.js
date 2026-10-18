// Library button "Pictures -> PDF": pick pictures, build one PDF (page per picture), save it to the library and open it.
import { pickImages } from '../io/pick-images.js';
import { imagesToPdf } from '../io/images-to-pdf.js';
import { openFile } from '../app/open-file.js';
import { toast } from './toast.js';
import { t } from '../core/i18n.js';
const btn = document.getElementById('images-btn');
const stamp = () => new Date().toISOString().slice(0, 16).replace('T', ' ').replace(':', '-');
btn.addEventListener('click', async () => {
  const files = await pickImages();
  if (!files.length) return;
  const name = (prompt(t('img.name'), `${t('img.defaultName')} ${stamp()}`) ?? '').trim();
  if (!name) return;
  btn.disabled = true;
  try {
    const blob = await imagesToPdf(files, (i, n) => toast(t('img.progress').replace('{i}', i).replace('{n}', n)));
    await openFile(new File([blob], name.replace(/\.pdf$/i, '') + '.pdf', { type: 'application/pdf' }));
  } catch { toast(t('img.fail')); }
  btn.disabled = false;
});
