const STYLE = `
<style>
  :host { display: block; width: 100%; }

  input {
    display: block;
    width: 100%;
    padding: 0.45em 0.9em;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface-strong);
    color: var(--text);
    font: inherit;
    font-size: var(--text-sm, 0.875rem);
    min-height: var(--touch-min, 44px);
    box-sizing: border-box;
    -webkit-appearance: none;
    appearance: none;
    transition: border-color 0.2s, background 0.2s;
  }

  input:focus {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-color: var(--accent);
    background: var(--surface);
  }

  input[type="search"]::-webkit-search-cancel-button { -webkit-appearance: none; }
</style>
`;

class SiteSearch extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = STYLE + `<input type="search" autocomplete="off" autocorrect="off" spellcheck="false">`;
    this.shadowRoot.querySelector('input').addEventListener('input', e =>
      this.dispatchEvent(new CustomEvent('search', { detail: e.target.value, bubbles: true, composed: true }))
    );
  }

  static get observedAttributes() { return ['placeholder']; }
  attributeChangedCallback(n, _, v) {
    if (n === 'placeholder') this.shadowRoot.querySelector('input').placeholder = v;
  }
}

customElements.define('site-search', SiteSearch);
