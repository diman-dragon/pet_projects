import fs from 'node:fs';
import path from 'node:path';

const ROOT = process.cwd();
const EXPECTED_DOCUMENT_COUNT = 144;
const ALLOWED_UNLISTED_FILES = new Set();

const errors = [];

function readJson(file) {
  return JSON.parse(fs.readFileSync(path.join(ROOT, file), 'utf8'));
}

function existsLocal(ref) {
  const clean = ref.replace(/^[.][/\\]/, '').split(/[?#]/)[0];
  return fs.existsSync(path.join(ROOT, clean));
}

function walk(dir) {
  const full = path.join(ROOT, dir);
  if (!fs.existsSync(full)) return [];
  return fs.readdirSync(full, { withFileTypes: true }).flatMap((entry) => {
    const rel = path.join(dir, entry.name).replaceAll('\\', '/');
    return entry.isDirectory() ? walk(rel) : [rel];
  });
}

function collectCatalogDocs() {
  const index = readJson('scripts/data/documents.json');
  const docs = [];
  for (const file of index.categoryFiles) {
    const category = readJson(`scripts/data/catalog/${file}`);
    const buckets = category.subcategories || [{ id: null, documents: category.documents || [] }];
    for (const sub of buckets) {
      for (const doc of sub.documents || []) {
        docs.push({
          ...doc,
          categoryId: category.id,
          subcategoryId: sub.id || null,
          fileBasePath: category.fileBasePath || category.id,
          fileSubpath: doc.fileSubpath || sub.fileSubpath || sub.id || null,
        });
      }
    }
  }
  return docs;
}

function resolveKey(obj, dottedKey) {
  return dottedKey.split('.').reduce((acc, part) => (acc && acc[part] !== undefined ? acc[part] : undefined), obj);
}

function isValidImage(file) {
  const buf = fs.readFileSync(path.join(ROOT, file));
  if (file.endsWith('.png')) return buf.subarray(0, 8).toString('hex') === '89504e470d0a1a0a';
  if (file.endsWith('.jpg') || file.endsWith('.jpeg')) {
    return buf.length > 4 && buf[0] === 0xff && buf[1] === 0xd8 && buf[buf.length - 2] === 0xff && buf[buf.length - 1] === 0xd9;
  }
  if (file.endsWith('.svg')) return buf.toString('utf8', 0, 200).includes('<svg');
  return true;
}

function checkCatalog() {
  const docs = collectCatalogDocs();
  if (docs.length !== EXPECTED_DOCUMENT_COUNT) {
    errors.push(`Document count is ${docs.length}, expected ${EXPECTED_DOCUMENT_COUNT}.`);
  }

  const ids = new Set();
  for (const doc of docs) {
    if (ids.has(doc.id)) errors.push(`Duplicate document id: ${doc.id}`);
    ids.add(doc.id);

    const rel = ['files', doc.fileBasePath || doc.categoryId, doc.fileSubpath || doc.subcategoryId, doc.filename].filter(Boolean).join('/');
    if (!existsLocal(rel)) errors.push(`Catalog file is missing: ${rel}`);
  }

  for (const doc of docs) {
    if (doc.threadParentId && !ids.has(doc.threadParentId)) {
      errors.push(`Missing thread parent for ${doc.id}: ${doc.threadParentId}`);
    }
  }

  const listedFiles = new Set(docs.map((doc) => ['files', doc.fileBasePath || doc.categoryId, doc.fileSubpath || doc.subcategoryId, doc.filename].filter(Boolean).join('/')));
  for (const file of walk('files')) {
    if (!listedFiles.has(file) && !ALLOWED_UNLISTED_FILES.has(file)) {
      errors.push(`File exists but is not in the document catalog: ${file}`);
    }
  }

  for (const lang of ['ru', 'en', 'sr']) {
    const i18n = readJson(`scripts/data/i18n/docs/${lang}/documents.json`);
    for (const doc of docs) {
      const key = doc.title_i18n_key.replace(/^doc[.]/, '');
      if (!resolveKey(i18n, key)) {
        errors.push(`Missing ${lang} document i18n for ${doc.id}: ${doc.title_i18n_key}`);
      }
    }
  }

  return docs;
}

function checkPersons(docs) {
  const ids = new Set(docs.map((doc) => doc.id));
  for (const lang of ['ru', 'en', 'sr']) {
    const people = readJson(`scripts/data/i18n/persons/${lang}.json`);
    for (const item of people.profiles.items || []) {
      if (item.photo?.startsWith('./')) {
        if (!existsLocal(item.photo)) errors.push(`Missing local person photo (${lang}/${item.id}): ${item.photo}`);
        else if (!isValidImage(item.photo.replace(/^[.]\//, ''))) errors.push(`Invalid local person photo (${lang}/${item.id}): ${item.photo}`);
      }
      for (const docId of item.docIds || []) {
        if (!ids.has(docId)) errors.push(`Unknown person docId (${lang}/${item.id}): ${docId}`);
      }
    }
  }
}

function collectLocalRefsFromJson(value, refs = []) {
  if (Array.isArray(value)) {
    value.forEach((item) => collectLocalRefsFromJson(item, refs));
  } else if (value && typeof value === 'object') {
    Object.values(value).forEach((item) => collectLocalRefsFromJson(item, refs));
  } else if (typeof value === 'string') {
    const attrRefs = [...value.matchAll(/\b(?:href|src)=["'](\.\/[^"']+)["']/g)].map((match) => match[1]);
    refs.push(...attrRefs);
    if (value.startsWith('./')) refs.push(value);
  }
  return refs;
}

function checkLocalLinks() {
  const filesToScan = [
    'index.html',
    ...walk('scripts/data').filter((file) => file.endsWith('.json')),
  ];

  for (const file of filesToScan) {
    const full = path.join(ROOT, file);
    const refs = file.endsWith('.json')
      ? collectLocalRefsFromJson(JSON.parse(fs.readFileSync(full, 'utf8')))
      : [...fs.readFileSync(full, 'utf8').matchAll(/\b(?:href|src|content)=["'](\.\/[^"']+)["']/g)].map((match) => match[1]);

    for (const ref of refs) {
      if (ref.includes('${') || ref.startsWith('./scripts/data/i18n/')) continue;
      if (!existsLocal(ref)) errors.push(`Broken local reference in ${file}: ${ref}`);
    }
  }
}

const docs = checkCatalog();
checkPersons(docs);
checkLocalLinks();

if (errors.length) {
  console.error(errors.join('\n'));
  process.exit(1);
}

console.log(`Site verification passed: ${docs.length} catalog documents, local links and person references are consistent.`);
