import { store } from '../store.js';
import { NAV_GROUPS, getGroupLabels } from './nav-config.js';
import './win11/ui-controls.js';
import './site-search.js';

const TMPL = `
<div class="hdr-shell">
  <div class="hdr-meta">
    <span class="hdr-meta__eyebrow" id="hdr-meta-eyebrow"></span>
    <span class="hdr-meta__summary" id="hdr-meta-summary"></span>
  </div>

  <div class="hdr-bar">
    <div class="hdr-brand" role="link" tabindex="0" aria-label="">
      <img src="./logo.png" alt="">
      <div class="hdr-brand__text">
        <span class="hdr-brand__name" id="hdr-brand-name"></span>
        <span class="hdr-brand__tagline" id="hdr-brand-tagline"></span>
      </div>
    </div>

    <div class="hdr-center">
      <nav class="hdr-desktop" id="hdr-desktop-nav"></nav>
    </div>

    <div class="hdr-utility">
      <div class="hdr-search">
        <site-search data-slot="search-desk"></site-search>
      </div>
      <ui-controls data-slot="ctrl-desk"></ui-controls>
    </div>

    <button class="hdr-hamburger" aria-label="" aria-expanded="false" aria-controls="hdr-drawer">
      <span></span><span></span><span></span>
    </button>
  </div>

  <div class="hdr-drawer" id="hdr-drawer" role="navigation" aria-label="">
    <div class="hdr-drawer__nav" id="hdr-drawer-nav"></div>
    <div class="hdr-drawer__search">
      <site-search data-slot="search-mob"></site-search>
    </div>
    <div class="hdr-drawer__controls">
      <ui-controls data-slot="ctrl-mob"></ui-controls>
    </div>
  </div>
</div>
`;

class SiteHeader extends HTMLElement {
  connectedCallback() {
    if (this._init) return;
    this._init = true;
    this.innerHTML = TMPL;

    this._drawer = this.querySelector('.hdr-drawer');
    this._burger = this.querySelector('.hdr-hamburger');
    this._desktopNav = this.querySelector('#hdr-desktop-nav');
    this._menuOpen = false;
    this._openGroup = null;
    this._navText = {};
    this._observer = null;
    this._revealTimer = null;

    const brand = this.querySelector('.hdr-brand');
    const goHome = () => this._navigateToPage('home');
    brand.addEventListener('click', goHome);
    brand.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        goHome();
      }
    });

    this._burger.addEventListener('click', () => this._toggleMenu());

    this._outside = (e) => {
      if (!this.contains(e.target)) {
        this._closeMenu();
        this._closeGroup();
      }
    };
    this._onEscape = (e) => {
      if (e.key === 'Escape') {
        this._closeMenu();
        this._closeGroup();
      }
    };

    this._desktopNav?.addEventListener('click', (e) => {
      const navButton = e.target.closest('[data-nav-id]');
      if (navButton) {
        e.stopPropagation();
        this._navigateToPage(navButton.dataset.navId);
        return;
      }

      const trigger = e.target.closest('[data-group-trigger]');
      if (!trigger) return;
      e.stopPropagation();
      const groupId = trigger.dataset.groupTrigger;
      this._toggleGroup(groupId);
    });

    this.querySelector('#hdr-drawer-nav')?.addEventListener('click', (e) => {
      const button = e.target.closest('[data-nav-id]');
      if (!button) return;
      e.stopPropagation();
      this._navigateToPage(button.dataset.navId);
    });

    this.querySelectorAll('site-search').forEach((search) => {
      search.addEventListener('search', (e) => {
        this.dispatchEvent(new CustomEvent('search', { detail: e.detail, bubbles: true, composed: true }));
      });
    });

    this._loadI18n();
    this._observeVisibility();

    let prevLang = store.state.lang;
    let prevPage = store.state.activePage;
    this._unsub = store.subscribe((state) => {
      if (state.lang !== prevLang) {
        prevLang = state.lang;
        this._loadI18n();
      }
      if (state.activePage !== prevPage) {
        prevPage = state.activePage;
        this._renderDesktopNav();
        this._renderDrawerNav();
      }
    });
  }

  disconnectedCallback() {
    if (this._unsub) this._unsub();
    if (this._observer) this._observer.disconnect();
    if (this._revealTimer) clearTimeout(this._revealTimer);
    document.removeEventListener('click', this._outside);
    document.removeEventListener('keydown', this._onEscape);
  }

  _emitNavigate(pageId) {
    this.dispatchEvent(new CustomEvent('navigate', { detail: pageId, bubbles: true, composed: true }));
  }

  _navigateToPage(pageId) {
    if (!pageId) return;
    this._emitNavigate(pageId);
    this._closeMenu();
    this._closeGroup();
  }

  _toggleMenu() {
    this._menuOpen ? this._closeMenu() : this._openMenu();
  }

  _openMenu() {
    this._menuOpen = true;
    this._drawer.classList.add('open');
    this._burger.setAttribute('aria-expanded', 'true');
    document.addEventListener('click', this._outside, { passive: true });
    document.addEventListener('keydown', this._onEscape);
  }

  _closeMenu() {
    this._menuOpen = false;
    this._drawer.classList.remove('open');
    this._burger.setAttribute('aria-expanded', 'false');
    document.removeEventListener('click', this._outside);
    document.removeEventListener('keydown', this._onEscape);
  }

  _toggleGroup(groupId) {
    this._openGroup = this._openGroup === groupId ? null : groupId;
    this._renderDesktopNav();
    if (this._openGroup) {
      document.addEventListener('click', this._outside, { passive: true });
      document.addEventListener('keydown', this._onEscape);
    }
  }

  _closeGroup() {
    if (!this._openGroup) return;
    this._openGroup = null;
    this._renderDesktopNav();
    document.removeEventListener('click', this._outside);
    document.removeEventListener('keydown', this._onEscape);
  }

  _observeVisibility() {
    this._observer = new IntersectionObserver((entries) => {
      const entry = entries[0];
      if (!entry?.isIntersecting) return;
      if ((window.scrollY || 0) < 24) return;
      this.classList.remove('is-revealing');
      if (this._revealTimer) clearTimeout(this._revealTimer);
      this._revealTimer = window.setTimeout(() => {
        this.classList.add('is-revealing');
        this._revealTimer = window.setTimeout(() => this.classList.remove('is-revealing'), 460);
      }, 20);
    }, { threshold: 0.45 });
    this._observer.observe(this);
  }

  async _loadI18n() {
    const lang = store.state.lang;

    try {
      let headerRes = await fetch(`./scripts/data/i18n/header/${lang}.json`);
      if (!headerRes.ok) headerRes = await fetch('./scripts/data/i18n/header/ru.json');
      const headerText = await headerRes.json();
      const ui = headerText.ui || {};

      const brand = this.querySelector('.hdr-brand');
      if (brand) {
        brand.setAttribute('aria-label', ui.brand_label || '');
        const img = brand.querySelector('img');
        if (img) img.alt = ui.brand_label || '';
      }
      if (this._burger) this._burger.setAttribute('aria-label', ui.menu_label || '');
      if (this._drawer) this._drawer.setAttribute('aria-label', ui.mobnav_label || '');

      const eyebrowEl = this.querySelector('#hdr-meta-eyebrow');
      const summaryEl = this.querySelector('#hdr-meta-summary');
      if (eyebrowEl) eyebrowEl.textContent = ui.eyebrow || '';
      if (summaryEl) summaryEl.textContent = ui.summary || '';

      this.querySelectorAll('site-search').forEach((s) => {
        s.setAttribute('placeholder', headerText.search_placeholder || '');
      });

      const nameEl = this.querySelector('#hdr-brand-name');
      const tagEl = this.querySelector('#hdr-brand-tagline');
      if (nameEl) nameEl.textContent = headerText.brand || '';
      if (tagEl) tagEl.textContent = headerText.tagline || '';

      let navRes = await fetch(`./scripts/data/i18n/nav/${lang}.json`);
      if (!navRes.ok) navRes = await fetch('./scripts/data/i18n/nav/ru.json');
      this._navText = await navRes.json();

      this._renderDesktopNav();
      this._renderDrawerNav();
    } catch (error) {
      console.error('Header i18n failed:', error);
    }
  }

  _renderDesktopNav() {
    const host = this._desktopNav;
    if (!host) return;

    const lang = store.state.lang;
    const labels = getGroupLabels(lang);
    const descriptions = this._navText.descriptions || {};
    const activePage = store.state.activePage || 'home';

    host.innerHTML = NAV_GROUPS.map((group) => {
      const isOpen = this._openGroup === group.id;
      return `
        <div class="hdr-group ${isOpen ? 'is-open' : ''}">
          <button type="button" class="hdr-group__trigger" data-group-trigger="${group.id}" aria-expanded="${isOpen ? 'true' : 'false'}">
            <span>${labels[group.id] || group.id}</span>
          </button>
          <div class="hdr-group__panel">
            ${group.pages.map((pageId) => `
              <button type="button" class="hdr-group__link ${pageId === activePage ? 'is-active' : ''}" data-nav-id="${pageId}">
                <span class="hdr-group__link-title">${this._navText[pageId] || pageId}</span>
                <span class="hdr-group__link-desc">${descriptions[pageId] || ''}</span>
              </button>
            `).join('')}
          </div>
        </div>
      `;
    }).join('');
  }

  _renderDrawerNav() {
    const host = this.querySelector('#hdr-drawer-nav');
    if (!host) return;

    const lang = store.state.lang;
    const labels = getGroupLabels(lang);
    const activePage = store.state.activePage || 'home';

    host.innerHTML = NAV_GROUPS.map((group) => `
      <section class="hdr-drawer__group">
        <div class="hdr-drawer__group-title">${labels[group.id] || group.id}</div>
        <div class="hdr-drawer__group-grid">
          ${group.pages.map((pageId) => `
            <button type="button" class="hdr-drawer__link ${pageId === activePage ? 'is-active' : ''}" data-nav-id="${pageId}">
              ${this._navText[pageId] || pageId}
            </button>
          `).join('')}
        </div>
      </section>
    `).join('');
  }
}

customElements.define('site-header', SiteHeader);

