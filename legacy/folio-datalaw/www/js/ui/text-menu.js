// Bubble menu over the selection (Copy / Highlight / Underline): positioning and button events.
const menu = document.getElementById('text-menu');
export const hideMenu = () => { menu.hidden = true; };
export function onMenuAction(fn) {
  menu.addEventListener('click', (e) => { const act = e.target.closest('button')?.dataset.act; if (act) fn(act); });
  menu.addEventListener('pointerdown', (e) => e.stopPropagation());
}
// anchor: selection box in viewport px; bounds: visible viewer area.
export function showMenu(anchor, bounds) {
  menu.hidden = false;
  const mw = menu.offsetWidth, mh = menu.offsetHeight;
  const x = Math.min(innerWidth - mw / 2 - 8, Math.max(mw / 2 + 8, (anchor.left + anchor.right) / 2));
  const below = anchor.top - mh - 14 < bounds.top;
  const y = below ? Math.min(anchor.bottom + 40, bounds.bottom - mh - 8) : anchor.top - 14;
  menu.classList.toggle('below', below);
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
}
