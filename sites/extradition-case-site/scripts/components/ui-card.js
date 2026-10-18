const template = document.createElement('template');
template.innerHTML = `
<style>
  .ui-card {
    display: block;
    height: 100%;
    background:
      linear-gradient(180deg, rgba(255,255,255,0.025), transparent 26%),
      var(--surface-strong);
    border: 1px solid var(--border);
    border-radius: 1.5rem;
    padding: 1.5rem;
    color: var(--text);
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
  }
  @media (hover: hover) and (pointer: fine) {
    .ui-card:hover {
      border-color: color-mix(in srgb, var(--accent) 46%, var(--border));
      box-shadow: 0 10px 26px rgba(0,0,0,0.16);
      transform: translateY(-2px);
    }
  }
  .ui-card__body {
    display: grid;
    gap: 0.8rem;
  }
  .ui-card__type {
    margin: 0;
    color: var(--text-faint);
    font-size: var(--text-xs, 0.75rem);
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .ui-card__title {
    margin: 0;
    font-size: var(--text-lg, 1.25rem);
    line-height: 1.25;
  }
  .ui-card__content {
    color: var(--text-muted);
    font-size: var(--text-base, 1rem);
    line-height: 1.72;
  }
  .ui-card__content > *:first-child {
    margin-top: 0;
  }
  .ui-card__content > *:last-child {
    margin-bottom: 0;
  }
  .ui-card__content p,
  .ui-card__content ul,
  .ui-card__content ol {
    margin: 0;
  }
  .ui-card__content ul,
  .ui-card__content ol {
    display: grid;
    gap: 0.55rem;
    padding-left: 1.1rem;
  }
  .ui-card__content a {
    color: var(--accent);
    text-decoration: none;
    font-weight: 600;
  }
  .ui-card__content a:hover {
    text-decoration: underline;
  }
  .primary, .secondary {
    display: inline-block;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.75rem 1rem;
    cursor: pointer;
    text-decoration: none;
    font-size: 0.9rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  .primary {
    background: var(--accent);
    color: var(--accent-soft);
  }
  .secondary {
    background: var(--surface-strong);
    color: var(--text);
  }
  .primary:hover, .secondary:hover {
    border-color: var(--accent);
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
  }
</style>
<article class="ui-card">
  <div class="ui-card__body">
    <p class="ui-card__type"></p>
    <h3 class="ui-card__title"></h3>
    <div class="ui-card__content"></div>
  </div>
</article>
`;

class UICard extends HTMLElement {
  constructor() {
    super();
    this.appendChild(template.content.cloneNode(true));
    this.data = null;
    this.lang = 'ru';
    this.titleEl   = this.querySelector('.ui-card__title');
    this.typeEl    = this.querySelector('.ui-card__type');
    this.contentEl = this.querySelector('.ui-card__content');
  }

  connectedCallback() {
    this.render();
  }

  setContent(data, lang = this.lang) {
    this.data = data;
    this.lang = lang;
    this.render();
  }

  render() {
    if (!this.data) {
      this.titleEl.textContent   = '';
      this.typeEl.textContent    = '';
      this.contentEl.innerHTML   = '';
      return;
    }

    // Support both direct strings and language-keyed objects
    const resolve = (val) => {
      if (!val) return '';
      if (typeof val === 'string') return val;
      return val[this.lang] ?? val['ru'] ?? val['en'] ?? '';
    };

    const title = resolve(this.data.title);
    const text  = resolve(this.data.text);

    this.titleEl.textContent = title;
    this.typeEl.textContent  = resolve(this.data.type);
    this.contentEl.innerHTML = text;

    this.classList.toggle('ui-card--feature',   this.data.type === 'feature');
    this.classList.toggle('ui-card--highlight',  this.data.type === 'highlight');
    this.classList.toggle('ui-card--entry',      this.data.type === 'entry');
    this.classList.toggle('ui-card--news',       this.data.type === 'news');
  }
}

customElements.define('ui-card', UICard);
