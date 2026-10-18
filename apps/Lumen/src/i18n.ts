type Dict = Record<string, string>;

const ru: Dict = {
  photos: 'Фото',
  albums: 'Альбомы',
  folders: 'Папки',
  favorites: 'Избранное',
  trash: 'Корзина',
  settings: 'Настройки',
  search: 'Поиск по имени и папке',
  all: 'Всё',
  onlyPhotos: 'Фото',
  onlyVideos: 'Видео',
  today: 'Сегодня',
  yesterday: 'Вчера',
  items: '{n} файлов',
  item1: '1 файл',
  selected: 'Выбрано: {n}',
  selectAll: 'Выбрать всё',
  share: 'Поделиться',
  move: 'Переместить',
  copy: 'Копировать',
  delete: 'Удалить',
  rename: 'Переименовать',
  info: 'Информация',
  openWith: 'Открыть в…',
  favorite: 'В избранное',
  unfavorite: 'Убрать из избранного',
  hideFolder: 'Скрыть папку',
  cancel: 'Отмена',
  ok: 'ОК',
  create: 'Создать',
  newFolder: 'Новая папка',
  folderName: 'Название папки',
  pickFolder: 'Куда?',
  pickFolderIn: 'Создать в',
  moveDone: 'Перемещено: {n}',
  copyDone: 'Скопировано: {n}',
  opFailed: 'Не удалось: {n}',
  trashed: 'В корзине: {n}',
  restored: 'Восстановлено: {n}',
  deletedForever: 'Удалено навсегда: {n}',
  confirmTrashTitle: 'Удалить {n}?',
  confirmTrashText: 'Файлы попадут в корзину и будут храниться 30 дней.',
  confirmForeverTitle: 'Удалить навсегда?',
  confirmForeverText: 'Это действие нельзя отменить.',
  restore: 'Восстановить',
  deleteForever: 'Удалить навсегда',
  emptyLibrary: 'Здесь пока пусто',
  emptyFav: 'Нажмите ★ на фото, чтобы добавить в избранное',
  emptyTrash: 'Корзина пуста',
  emptySearch: 'Ничего не найдено',
  noAccessTitle: 'Нужен доступ к фото',
  noAccessText: 'Lumen показывает фото и видео прямо из памяти устройства. Ничего не загружается в сеть.',
  grant: 'Разрешить доступ',
  openSettings: 'Если окно не появляется — разрешите доступ к фото и видео в настройках приложения.',
  scanning: 'Читаю библиотеку…',
  sort: 'Сортировка',
  sortDate: 'По дате',
  sortName: 'По имени',
  sortSize: 'По размеру',
  desc: 'Сначала новые / большие',
  asc: 'Сначала старые / маленькие',
  filter: 'Показывать',
  view: 'Вид',
  theme: 'Тема',
  themeAuto: 'Системная',
  themeDark: 'Тёмная',
  themeLight: 'Светлая',
  themeAmoled: 'AMOLED',
  columns: 'Колонок в сетке',
  groupByDay: 'Группировать по дням',
  fitPhotos: 'Показывать фото целиком',
  language: 'Язык',
  langAuto: 'Системный',
  hiddenFolders: 'Скрытые папки',
  unhide: 'Показать',
  noHidden: 'Нет скрытых папок',
  manageMedia: 'Операции без подтверждений',
  manageMediaHint: 'Разрешает перемещать и удалять в корзину без системного окна (Android 12+).',
  manageMediaOn: 'Включено',
  manageMediaOff: 'Включить',
  infoName: 'Имя',
  infoPath: 'Путь',
  infoSize: 'Размер',
  infoRes: 'Разрешение',
  infoDate: 'Дата съёмки',
  infoType: 'Тип',
  infoCamera: 'Камера',
  infoExposure: 'Экспозиция',
  infoGps: 'Координаты',
  infoDuration: 'Длительность',
  renameTitle: 'Переименовать',
  renamed: 'Переименовано',
  back: 'Назад',
  more: 'Ещё',
  root: 'Память',
  folderStat: '{c} · {s}',
  cancelled: 'Отменено',
  error: 'Ошибка',
  about: 'Lumen — быстрый просмотрщик фото и видео',
  close: 'Закрыть',
  playVideo: 'Видео',
};

const en: Dict = {
  photos: 'Photos',
  albums: 'Albums',
  folders: 'Folders',
  favorites: 'Favorites',
  trash: 'Trash',
  settings: 'Settings',
  search: 'Search by name or folder',
  all: 'All',
  onlyPhotos: 'Photos',
  onlyVideos: 'Videos',
  today: 'Today',
  yesterday: 'Yesterday',
  items: '{n} items',
  item1: '1 item',
  selected: '{n} selected',
  selectAll: 'Select all',
  share: 'Share',
  move: 'Move',
  copy: 'Copy',
  delete: 'Delete',
  rename: 'Rename',
  info: 'Info',
  openWith: 'Open with…',
  favorite: 'Add to favorites',
  unfavorite: 'Remove from favorites',
  hideFolder: 'Hide folder',
  cancel: 'Cancel',
  ok: 'OK',
  create: 'Create',
  newFolder: 'New folder',
  folderName: 'Folder name',
  pickFolder: 'Where to?',
  pickFolderIn: 'Create in',
  moveDone: 'Moved: {n}',
  copyDone: 'Copied: {n}',
  opFailed: 'Failed: {n}',
  trashed: 'Moved to trash: {n}',
  restored: 'Restored: {n}',
  deletedForever: 'Deleted forever: {n}',
  confirmTrashTitle: 'Delete {n}?',
  confirmTrashText: 'Files go to the trash and are kept for 30 days.',
  confirmForeverTitle: 'Delete forever?',
  confirmForeverText: 'This cannot be undone.',
  restore: 'Restore',
  deleteForever: 'Delete forever',
  emptyLibrary: 'Nothing here yet',
  emptyFav: 'Tap ★ on a photo to add it to favorites',
  emptyTrash: 'Trash is empty',
  emptySearch: 'Nothing found',
  noAccessTitle: 'Photo access needed',
  noAccessText: 'Lumen shows photos and videos straight from device storage. Nothing is uploaded.',
  grant: 'Allow access',
  openSettings: 'If no prompt appears, allow photo and video access in the app settings.',
  scanning: 'Reading library…',
  sort: 'Sort',
  sortDate: 'By date',
  sortName: 'By name',
  sortSize: 'By size',
  desc: 'Newest / largest first',
  asc: 'Oldest / smallest first',
  filter: 'Show',
  view: 'View',
  theme: 'Theme',
  themeAuto: 'System',
  themeDark: 'Dark',
  themeLight: 'Light',
  themeAmoled: 'AMOLED',
  columns: 'Grid columns',
  groupByDay: 'Group by day',
  fitPhotos: 'Show whole photos',
  language: 'Language',
  langAuto: 'System',
  hiddenFolders: 'Hidden folders',
  unhide: 'Show',
  noHidden: 'No hidden folders',
  manageMedia: 'Skip confirmation dialogs',
  manageMediaHint: 'Lets the app move and trash media without system prompts (Android 12+).',
  manageMediaOn: 'Enabled',
  manageMediaOff: 'Enable',
  infoName: 'Name',
  infoPath: 'Path',
  infoSize: 'Size',
  infoRes: 'Resolution',
  infoDate: 'Date taken',
  infoType: 'Type',
  infoCamera: 'Camera',
  infoExposure: 'Exposure',
  infoGps: 'Location',
  infoDuration: 'Duration',
  renameTitle: 'Rename',
  renamed: 'Renamed',
  back: 'Back',
  more: 'More',
  root: 'Storage',
  folderStat: '{c} · {s}',
  cancelled: 'Cancelled',
  error: 'Error',
  about: 'Lumen — a fast photo & video viewer',
  close: 'Close',
  playVideo: 'Video',
};

const dicts: Record<string, Dict> = { ru, en };
export let lang: 'ru' | 'en' = 'ru';

export function setLang(pref: 'auto' | 'ru' | 'en'): void {
  if (pref === 'auto') {
    const nav = (typeof navigator !== 'undefined' ? navigator.language : 'en').toLowerCase();
    lang = nav.startsWith('ru') || nav.startsWith('uk') || nav.startsWith('be') ? 'ru' : 'en';
  } else lang = pref;
  document.documentElement.lang = lang;
}

export function t(key: string, vars?: Record<string, string | number>): string {
  let s = dicts[lang][key] ?? ru[key] ?? key;
  if (vars) for (const k of Object.keys(vars)) s = s.replace(`{${k}}`, String(vars[k]));
  return s;
}

export function fmtCount(n: number): string {
  return n === 1 ? t('item1') : t('items', { n: n.toLocaleString(lang) });
}

export function fmtSize(b: number): string {
  if (b < 1024) return `${b} B`;
  const u = ['KB', 'MB', 'GB', 'TB'];
  let v = b / 1024;
  let i = 0;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${u[i]}`;
}

export function fmtDuration(ms: number): string {
  const s = Math.round(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}

export function fmtDay(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  const dayStart = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((dayStart(now) - dayStart(d)) / 86400000);
  if (diff === 0) return t('today');
  if (diff === 1) return t('yesterday');
  return new Intl.DateTimeFormat(lang, {
    weekday: 'short',
    day: 'numeric',
    month: 'long',
    ...(d.getFullYear() !== now.getFullYear() ? { year: 'numeric' as const } : {}),
  }).format(d);
}

export function fmtMonth(ts: number): string {
  return new Intl.DateTimeFormat(lang, { month: 'long', year: 'numeric' }).format(new Date(ts));
}

export function fmtDateTime(ts: number): string {
  return new Intl.DateTimeFormat(lang, { dateStyle: 'long', timeStyle: 'short' }).format(new Date(ts));
}
