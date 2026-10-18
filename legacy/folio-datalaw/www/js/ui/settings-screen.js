// Settings screen: segmented switches (theme, language, night, reading mode, keep-awake), folder row and the version row.
import { getTheme, setTheme } from '../core/theme.js';
import { getLang, setLang } from '../core/i18n.js';
import { getNight, setNight } from '../core/night.js';
import { getReadMode, setReadMode } from '../core/readmode.js';
import { getAwake, setAwake } from '../core/awake-pref.js';
import { VERSION } from '../core/version.js';
import { on } from '../core/bus.js';
import './folder-row.js';
const get = { theme: getTheme, lang: getLang, night: getNight, mode: getReadMode, awake: getAwake };
const set = { theme: setTheme, lang: setLang, night: setNight, mode: setReadMode, awake: setAwake };
const segs = [...document.querySelectorAll('.seg[data-setting]')];
const paint = () => segs.forEach((s) => [...s.children].forEach((b) => b.classList.toggle('on', b.dataset.value === get[s.dataset.setting]())));
segs.forEach((s) => s.addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  set[s.dataset.setting](b.dataset.value);
  paint();
}));
on('modechange', paint);
document.getElementById('app-version').textContent = VERSION;
paint();
