import { store } from '../store.js';
import { escapeHtml, safeTelegramUrl, safeEmail } from '../security.js';
import { NAV_GROUPS, getGroupLabels } from './nav-config.js';

const TEMPLATE = `
  <footer class="site-footer__inner">
    <div class="site-footer__grid">
      <section class="site-footer__about">
        <div class="site-footer__eyebrow">Extradition Case Archive</div>
        <h4 id="footer-about-title"></h4>
        <p id="footer-about-text"></p>
        <p class="site-footer__disclaimer" id="footer-about-disclaimer"></p>
      </section>

      <section class="site-footer__nav">
        <h4 id="footer-nav-title"></h4>
        <div class="site-footer__nav-groups" id="footer-nav-groups"></div>
      </section>

      <section class="site-footer__contact">
        <h4 id="footer-contact-title"></h4>
        <p id="footer-contact-press"></p>
        <p id="footer-contact-legal"></p>
        <p id="footer-contact-email"></p>
        <p id="footer-contact-phone"></p>
      </section>
    </div>

    <div class="site-footer__bottom">
      <p id="footer-copy-text"></p>
      <p id="footer-copy-location"></p>
    </div>
  </footer>
`;

class SiteFooter extends HTMLElement {
  connectedCallback() {
    if (this._init) return;
    this._init = true;
    this.innerHTML = TEMPLATE;

    this.update(store.state);
    let prevLang = store.state.lang;
    this._unsub = store.subscribe((state) => {
      if (state.lang !== prevLang) {
        prevLang = state.lang;
        this.update(state);
      }
    });
  }

  disconnectedCallback() {
    if (this._unsub) this._unsub();
  }


  _navigateToPage(pageId) {
    if (!pageId) return;
    document.dispatchEvent(new CustomEvent('app:navigate', { detail: pageId }));
  }

  _renderNav(lang, navText) {
    const labels = getGroupLabels(lang);
    const host = this.querySelector('#footer-nav-groups');
    host.innerHTML = NAV_GROUPS.map((group) => `
      <section class="site-footer__nav-group">
        <div class="site-footer__nav-label">${escapeHtml(labels[group.id] || group.id)}</div>
        <div class="site-footer__nav-links">
          ${group.pages.map((id) => `
            <button type="button" class="site-footer__nav-link" data-id="${escapeHtml(id)}">${escapeHtml(navText[id] || id)}</button>
          `).join('')}
        </div>
      </section>
    `).join('');

    host.querySelectorAll('[data-id]').forEach((button) => {
      button.addEventListener('click', () => this._navigateToPage(button.dataset.id));
    });
  }

  _renderEmail(email) {
    const emailEl = this.querySelector('#footer-contact-email');
    const safeMailHref = safeEmail(email);
    emailEl.innerHTML = '';
    if (safeMailHref) {
      const a = document.createElement('a');
      a.href = `mailto:${safeMailHref}`;
      a.textContent = email;
      emailEl.appendChild(a);
      return;
    }
    emailEl.textContent = email || '';
  }

  _renderPhone(contact) {
    const phoneEl = this.querySelector('#footer-contact-phone');
    phoneEl.innerHTML = '';
    if (contact.phone) {
      phoneEl.appendChild(document.createTextNode(contact.phone));
      phoneEl.appendChild(document.createElement('br'));
    }

    const tgHref = safeTelegramUrl(contact.telegram_link);
    if (tgHref !== '#') {
      const tgA = document.createElement('a');
      tgA.href = tgHref;
      tgA.target = '_blank';
      tgA.rel = 'noopener noreferrer';
      tgA.textContent = contact.telegram || '';
      phoneEl.appendChild(tgA);
    } else {
      phoneEl.appendChild(document.createTextNode(contact.telegram || ''));
    }
  }

  async update(state) {
    try {
      let footerRes = await fetch(`./scripts/data/i18n/footer/${state.lang}.json`);
      if (!footerRes.ok) footerRes = await fetch('./scripts/data/i18n/footer/ru.json');
      let navRes = await fetch(`./scripts/data/i18n/nav/${state.lang}.json`);
      if (!navRes.ok) navRes = await fetch('./scripts/data/i18n/nav/ru.json');

      const t = await footerRes.json();
      const nav = await navRes.json();

      this.querySelector('#footer-about-title').textContent = t.about.title;
      this.querySelector('#footer-about-text').textContent = t.about.text;
      this.querySelector('#footer-about-disclaimer').textContent = t.about.disclaimer;

      this.querySelector('#footer-nav-title').textContent = t.nav.title;
      this._renderNav(state.lang, nav);

      this.querySelector('#footer-contact-title').textContent = t.contact.title;
      this.querySelector('#footer-contact-press').textContent = t.contact.press;
      this.querySelector('#footer-contact-legal').textContent = t.contact.legal;
      this._renderEmail(t.contact.email);
      this._renderPhone(t.contact);

      this.querySelector('#footer-copy-text').textContent = t.copyright.text;
      this.querySelector('#footer-copy-location').textContent = t.copyright.location;
    } catch (error) {
      console.error('Footer update failed:', error);
    }
  }
}

customElements.define('site-footer', SiteFooter);
