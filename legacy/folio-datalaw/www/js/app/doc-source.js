// Where a document's bytes come from: the library store (opened files) or the chosen folder (referenced by uri).
import { getBlob } from '../data/library.js';
import { readFolderFile } from '../io/folder.js';
export const loadBlob = (meta) => (meta.uri ? readFolderFile(meta.uri) : getBlob(meta.id));
