// Opens the system file chooser; resolves with the chosen File or null.
import { acceptList } from '../formats/registry.js';
export function pickFile() {
  return new Promise((res) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = acceptList();
    input.onchange = () => res(input.files[0] ?? null);
    input.oncancel = () => res(null);
    input.click();
  });
}
