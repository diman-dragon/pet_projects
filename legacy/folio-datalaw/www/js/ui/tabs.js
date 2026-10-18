// Bottom tab bar + swipe pager: screens sit side by side in a snap-scroll strip; swipe or tap switches, Back returns Home.
import { setBackFallback } from './back.js';
import { plugin } from '../core/platform.js';
const strip = document.getElementById('screens');
const screens = [...document.querySelectorAll('.screen')];
const btns = [...document.querySelectorAll('#tabbar button')];
let current = 0, raf = 0, lockUntil = 0;
const paint = (i) => { current = i; btns.forEach((b, k) => b.classList.toggle('on', k === i)); };
export function goTo(name, smooth = true) {
  const i = Math.max(0, screens.findIndex((s) => s.dataset.screen === name));
  lockUntil = Date.now() + 450;
  strip.scrollTo({ left: i * strip.clientWidth, behavior: smooth ? 'smooth' : 'instant' });
  paint(i);
}
strip.addEventListener('scroll', () => {
  if (Date.now() < lockUntil) return;
  cancelAnimationFrame(raf);
  raf = requestAnimationFrame(() => { const i = Math.round(strip.scrollLeft / (strip.clientWidth || 1)); if (i !== current) paint(i); });
}, { passive: true });
new ResizeObserver(() => { strip.scrollLeft = current * strip.clientWidth; }).observe(strip);
btns.forEach((b) => b.addEventListener('click', () => goTo(b.dataset.tab)));
setBackFallback(() => (current !== 0 ? goTo('home') : plugin('App')?.exitApp()));
paint(0);
