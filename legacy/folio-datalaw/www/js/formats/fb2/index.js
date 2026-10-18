// FictionBook (FB2, FB2.zip) format descriptor for the registry.
import { mountFb2 } from './mount.js';
import { fb2Cover } from './cover.js';
export const fb2 = {
  id: 'fb2',
  extensions: ['fb2', 'fb2.zip', 'fbz'],
  mimes: ['application/x-fictionbook+xml', 'application/fb2', 'application/x-fb2', 'application/fb2+zip'],
  mount: mountFb2,
  cover: fb2Cover,
};
