// Viewer night mode: .night class when dark theme + night setting on; toolbar toggle (dark theme only).
import { getTheme } from '../core/theme.js';
import { getNight, setNight } from '../core/night.js';
import { on } from '../core/bus.js';
const root = document.getElementById('viewer');
const btn = document.getElementById('night-btn');
function paint() {
  const dark = getTheme() === 'dark';
  root.classList.toggle('night', dark && getNight() === 'on');
  btn.hidden = !dark;
}
btn.addEventListener('click', () => setNight(getNight() === 'on' ? 'off' : 'on'));
on('themechange', paint);
on('nightchange', paint);
