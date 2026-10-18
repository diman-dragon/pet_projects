// Stylus mode (explicit toggle in the palette): only the pen draws, the pen never scrolls, fingers only scroll / zoom.
import { getPref, setPref } from '../core/prefs.js';
import { emit } from '../core/bus.js';
import { toast } from './toast.js';
import { t } from '../core/i18n.js';
const root = document.getElementById('viewer');
let only = getPref('stylus', false), hinted = false;
const paint = () => root.classList.toggle('stylus-on', only);
paint();
export const stylusOnly = () => only;
export function setStylusOnly(v) { only = v; setPref('stylus', v); paint(); emit('stylus', v); }
// First pen contact while the mode is off: tell the reader how to get "pen draws, fingers scroll".
export function penSeen() {
  if (only || hinted) return;
  hinted = true;
  toast(t('edit.stylusHint'));
}
