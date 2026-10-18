export const fullTextStyles = `
  .fts-mark {
    background: #fef08a;
    color: #854d0e;
    border-radius: 3px;
    padding: 0 2px;
    font-weight: 600;
  }

  .fts-fragment-panel .fts-mark {
    background: #fde68a;
    box-shadow: 0 0 0 1px rgba(245, 158, 11, 0.18);
  }

  .fts-citation {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 22px;
    height: 20px;
    border: none;
    border-radius: 6px;
    background: var(--app-accent, #2563eb);
    color: white;
    font-size: 10px;
    font-weight: 700;
    cursor: pointer;
    transition: transform 0.12s ease, opacity 0.12s ease;
  }

  .fts-citation:hover {
    opacity: 0.84;
    transform: scale(1.05);
  }
`;
