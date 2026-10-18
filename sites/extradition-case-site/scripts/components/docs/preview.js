import { escapeHtml, safeUrl } from '../../security.js';
import { buildFileUrl, fileTypeBadge } from './catalog.js';

const OVERLAY_ID = 'global-doc-preview-overlay';

function createOverlayShell() {
  const existing = document.getElementById(OVERLAY_ID);
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = OVERLAY_ID;
  overlay.className = 'doc-preview-overlay';
  overlay.hidden = true;
  overlay.innerHTML = `
    <div class="doc-preview-modal" role="dialog" aria-modal="true" aria-labelledby="preview-title">
      <div class="doc-preview-modal__head">
        <div class="doc-preview-modal__titles">
          <span id="preview-type-badge" class="docs-card__badge docs-card__badge--type"></span>
          <strong id="preview-title"></strong>
        </div>
        <div class="doc-preview-modal__actions">
          <a id="preview-download-btn" href="#" download class="docs-card__btn docs-card__btn--secondary"></a>
          <button type="button" id="preview-close-btn" class="docs-card__btn docs-card__btn--secondary"></button>
        </div>
      </div>
      <div id="preview-body" class="doc-preview-modal__body"></div>
    </div>
  `;

  document.body.appendChild(overlay);
  return overlay;
}

export function createDocumentPreview({ ui }) {
  const overlay = createOverlayShell();
  const previewTitle = overlay.querySelector('#preview-title');
  const previewBadge = overlay.querySelector('#preview-type-badge');
  const previewDlBtn = overlay.querySelector('#preview-download-btn');
  const previewBody = overlay.querySelector('#preview-body');
  const previewCloseBtn = overlay.querySelector('#preview-close-btn');

  function closePreview() {
    overlay.hidden = true;
    document.body.style.overflow = '';
    previewBody.innerHTML = '';
  }

  function openPreview(docRow, meta, type) {
    const fileUrl = buildFileUrl(docRow);

    previewTitle.textContent = meta.title;
    previewBadge.textContent = fileTypeBadge(type, ui);
    previewDlBtn.href = safeUrl(fileUrl);
    previewDlBtn.setAttribute('download', docRow.filename);
    previewDlBtn.textContent = `↓ ${ui.download}`;
    previewDlBtn.onclick = () => {
      if (typeof gtag === 'function') {
        gtag('event', 'file_download', {
          file_name: docRow.filename,
          file_extension: docRow.filename.split('.').pop(),
          link_text: meta?.title || docRow.filename,
          link_url: fileUrl,
          source: 'preview_modal',
        });
      }
    };
    previewCloseBtn.textContent = `${ui.close} ✕`;
    overlay.hidden = false;
    document.body.style.overflow = 'hidden';

    if (type === 'pdf') {
      previewBody.innerHTML = `<iframe src="${safeUrl(fileUrl)}#toolbar=1&navpanes=0&scrollbar=1&page=1&zoom=page-fit" title="${escapeHtml(meta.title)}" class="doc-preview-iframe"></iframe>`;
      return;
    }

    if (type === 'word') {
      const isLocal = /^https?:\/\/(localhost|127\.|0\.0\.0\.)/.test(window.location.origin);
      if (!isLocal) {
        const basePath = window.location.pathname.replace(/\/[^/]*$/, '');
        const absUrl = `${window.location.origin}${basePath}/${buildFileUrl(docRow).replace(/^[.]\//, '')}`;
        const officeUrl = `https://view.officeapps.live.com/op/embed.aspx?src=${encodeURIComponent(absUrl)}`;
        previewBody.innerHTML = `<iframe src="${officeUrl}" title="${escapeHtml(meta.title)}" class="doc-preview-iframe"></iframe>`;
        return;
      }
      previewBody.innerHTML = `
        <div class="doc-preview-fallback">
          <p>${escapeHtml(ui.wordNote)}</p>
          <a href="${safeUrl(fileUrl)}" download="${escapeHtml(docRow.filename)}" class="docs-card__btn docs-card__btn--primary">↓ ${escapeHtml(ui.download)}</a>
        </div>
      `;
      return;
    }

    previewBody.innerHTML = `<p class="docs-empty"><a href="${safeUrl(fileUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(ui.openNew)} ↗</a></p>`;
  }

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) closePreview();
  });
  previewCloseBtn.addEventListener('click', closePreview);

  const onKeydown = (e) => {
    if (e.key === 'Escape' && !overlay.hidden) closePreview();
  };
  document.addEventListener('keydown', onKeydown);

  return {
    openPreview,
    closePreview,
    destroy() {
      document.removeEventListener('keydown', onKeydown);
      closePreview();
      overlay.remove();
    },
  };
}
