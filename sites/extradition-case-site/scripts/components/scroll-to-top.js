import { store } from '../store.js';

const LABELS = { ru: 'Наверх', en: 'Back to top', sr: 'Nazad na vrh' };

class ScrollToTop extends HTMLElement {
  connectedCallback() {
    if (this._init) return;
    this._init = true;

    this.innerHTML = `
<style>
  scroll-to-top {
    position: fixed;
    bottom: calc(1.25rem + var(--safe-bottom, 0px));
    right: calc(1.25rem + var(--safe-right, 0px));
    z-index: 999;
    pointer-events: none;
  }

  .stt {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: var(--accent);
    color: #fff;
    border: none;
    cursor: pointer;
    font-size: 1.25rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    pointer-events: auto;
    touch-action: manipulation;
    -webkit-tap-highlight-color: transparent;
    opacity: 0;
    transform: translateY(10px) scale(0.9);
    transition: opacity 0.25s ease, transform 0.25s ease, filter 0.2s ease;
  }

  .stt.vis {
    opacity: 1;
    transform: translateY(0) scale(1);
  }

  .stt:hover { filter: brightness(1.15); }

  @media (prefers-reduced-motion: reduce) {
    .stt { transition: opacity 0.1s; }
  }
</style>
<button class="stt" aria-label="Наверх">↑</button>`;

    this._btn = this.querySelector('.stt');
    this._btn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));

    this._onScroll = () => this._btn.classList.toggle('vis', window.scrollY > 300);
    window.addEventListener('scroll', this._onScroll, { passive: true });
    this._onScroll();

    this._setLabel(store.state.lang);
    let _pl = store.state.lang;
    this._unsub = store.subscribe(s => { if (s.lang !== _pl) { _pl = s.lang; this._setLabel(s.lang); } });
  }

  disconnectedCallback() {
    window.removeEventListener('scroll', this._onScroll);
    if (this._unsub) this._unsub();
  }

  _setLabel(lang) {
    const l = LABELS[lang] || LABELS.en;
    this._btn.setAttribute('aria-label', l);
    this._btn.title = l;
  }
}

customElements.define('scroll-to-top', ScrollToTop);
