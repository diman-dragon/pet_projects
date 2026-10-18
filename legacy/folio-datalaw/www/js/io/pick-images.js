// Opens the system picker for several pictures (gallery / camera); resolves with an array of Files (empty when cancelled).
export function pickImages() {
  return new Promise((res) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.multiple = true;
    input.onchange = () => res([...input.files]);
    input.oncancel = () => res([]);
    input.click();
  });
}
