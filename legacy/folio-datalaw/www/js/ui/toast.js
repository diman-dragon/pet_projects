// Transient toast message.
const el = document.getElementById('toast');
let timer;
export function toast(msg) {
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(timer);
  timer = setTimeout(() => el.classList.remove('show'), 2500);
}
