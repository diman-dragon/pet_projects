// Prefixed localStorage wrapper that never throws.
export const store = {
  get(k) { try { return localStorage.getItem('folio.' + k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem('folio.' + k, v); } catch { /* ignore */ } },
};
