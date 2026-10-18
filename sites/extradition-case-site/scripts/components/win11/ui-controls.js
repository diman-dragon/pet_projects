import { store } from '../../store.js';

const STYLE = `
<style>
  :host {
    display: inline-flex;
    flex-direction: column;
    gap: 4px;
    background: var(--surface-strong);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm, 10px);
    padding: 5px 7px;
  }

  .group { display: flex; align-items: center; gap: 6px; }

  .label {
    font-size: var(--text-xs, 0.7rem);
    color: var(--text-muted);
    min-width: 30px;
    text-transform: uppercase;
    user-select: none;
  }

  .pills { display: flex; gap: 2px; }

  .pill {
    background: transparent;
    border: none;
    color: var(--text);
    padding: 3px 7px;
    border-radius: 4px;
    cursor: pointer;
    font: inherit;
    font-size: var(--text-xs, 0.72rem);
    min-height: 32px;
    min-width: 32px;
    touch-action: manipulation;
    -webkit-tap-highlight-color: transparent;
    transition: background 0.15s;
  }

  .pill[aria-pressed="true"] {
    background: var(--accent);
    color: var(--accent-soft);
    font-weight: 600;
  }

  .pill:hover:not([aria-pressed="true"]) { background: var(--border); }
</style>
`;

class UiControls extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = STYLE + `
      <div class="group">
        <span class="label" id="ll"></span>
        <div class="pills">
          <button class="pill" data-lang="ru">RU</button>
          <button class="pill" data-lang="sr">SR</button>
          <button class="pill" data-lang="en">EN</button>
        </div>
      </div>
      <div class="group">
        <span class="label" id="tl"></span>
        <div class="pills">
          <button class="pill" data-theme="dark"  id="td"></button>
          <button class="pill" data-theme="light" id="tl2"></button>
        </div>
      </div>
    `;
  }

  async connectedCallback() {
    this._updateActive('lang',  store.state.lang);
    this._updateActive('theme', store.state.theme);
    await this._renderLabels();

    let _pl = store.state.lang;
    this._unsub = store.subscribe(s => {
      this._updateActive('lang',  s.lang);
      this._updateActive('theme', s.theme);
      if (s.lang !== _pl) { _pl = s.lang; this._renderLabels(); }
    });

    this._initPills('lang',  v => store.setState({ lang: v }));
    this._initPills('theme', v => store.setState({ theme: v }));
  }

  disconnectedCallback() { if (this._unsub) this._unsub(); }

  async _renderLabels() {
    try {
      const r = await fetch(`./scripts/data/i18n/controls/${store.state.lang}.json`);
      if (!r.ok) return;
      const t = await r.json();
      this.shadowRoot.querySelector('#ll').textContent  = t.lang_label  || '';
      this.shadowRoot.querySelector('#tl').textContent  = t.theme_label || '';
      this.shadowRoot.querySelector('#td').textContent  = t.theme_dark  || '';
      this.shadowRoot.querySelector('#tl2').textContent = t.theme_light || '';
    } catch(e) {}
  }

  _initPills(type, cb) {
    this.shadowRoot.querySelectorAll(`[data-${type}]`).forEach(b =>
      b.addEventListener('click', () => cb(b.dataset[type]))
    );
  }

  _updateActive(type, val) {
    this.shadowRoot.querySelectorAll(`[data-${type}]`).forEach(b =>
      b.setAttribute('aria-pressed', String(b.dataset[type] === val))
    );
  }
}

customElements.define('ui-controls', UiControls);
