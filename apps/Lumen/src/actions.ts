import { api } from './native';
import { store } from './store';
import { thumbs } from './thumbs';
import { fmtCount, fmtDateTime, fmtDuration, fmtSize, t } from './i18n';
import { confirmDialog, h, haptic, icon, menu, promptDialog, sheet, toast } from './ui';
import type { Item, Ref } from './types';

const refs = (items: Item[]): Ref[] => items.map((i) => ({ id: i.id, v: i.v }));

function isCancel(e: unknown): boolean {
  const x = e as { code?: string; message?: string } | undefined;
  return x?.code === 'cancelled' || x?.message === 'cancelled';
}

function fail(e: unknown): void {
  if (isCancel(e)) return;
  const msg = (e as { message?: string } | undefined)?.message;
  toast(`${t('error')}${msg ? `: ${msg}` : ''}`, 3500);
}

export async function shareItems(items: Item[]): Promise<void> {
  try {
    await api.share({ items: refs(items) });
  } catch (e) {
    fail(e);
  }
}

export async function openWith(item: Item): Promise<void> {
  try {
    await api.openWith({ id: item.id, v: item.v });
  } catch (e) {
    fail(e);
  }
}

export function toggleFavorite(items: Item[]): void {
  const on = store.toggleFav(items.map((i) => i.k));
  haptic();
  toast(on ? t('favorite') : t('unfavorite'), 1400);
}

/** Удаление в корзину. Возвращает true, если файлы удалены. */
export async function trashItems(items: Item[]): Promise<boolean> {
  const ok = await confirmDialog(t('confirmTrashTitle', { n: fmtCount(items.length) }), t('confirmTrashText'), t('delete'), true);
  if (!ok) return false;
  try {
    const r = await api.trash({ items: refs(items) });
    if (r.done > 0) {
      store.removeKeys(items.map((i) => i.k));
      toast(t('trashed', { n: r.done }));
    }
    if (r.failed > 0) toast(t('opFailed', { n: r.failed }), 3500);
    void store.rescan();
    return r.done > 0;
  } catch (e) {
    fail(e);
    return false;
  }
}

export async function restoreItems(items: Item[]): Promise<boolean> {
  try {
    const r = await api.restore({ items: refs(items) });
    toast(t('restored', { n: r.done }));
    void store.rescan();
    return r.done > 0;
  } catch (e) {
    fail(e);
    return false;
  }
}

export async function deleteForever(items: Item[]): Promise<boolean> {
  const ok = await confirmDialog(t('confirmForeverTitle'), t('confirmForeverText'), t('deleteForever'), true);
  if (!ok) return false;
  try {
    const r = await api.deleteForever({ items: refs(items) });
    toast(t('deletedForever', { n: r.done }));
    return r.done > 0;
  } catch (e) {
    fail(e);
    return false;
  }
}

export async function moveItems(items: Item[], copy: boolean): Promise<boolean> {
  const dest = await pickFolder(items[0]?.dir);
  if (!dest) return false;
  try {
    const r = copy
      ? await api.copy({ items: refs(items), dest })
      : await api.move({ items: refs(items), dest });
    if (r.done > 0) toast(t(copy ? 'copyDone' : 'moveDone', { n: r.done }));
    if (r.failed > 0) toast(`${t('opFailed', { n: r.failed })}${r.error ? ` — ${r.error}` : ''}`, 4500);
    void store.rescan();
    return r.done > 0;
  } catch (e) {
    fail(e);
    return false;
  }
}

export async function renameItem(item: Item): Promise<boolean> {
  const raw = await promptDialog(t('renameTitle'), item.name, '', t('ok'));
  if (raw === null) return false;
  let name = raw.trim().replace(/[\\/:*?"<>|]/g, '_');
  if (!name || name === item.name) return false;
  const dot = item.name.lastIndexOf('.');
  if (dot > 0 && !name.includes('.')) name += item.name.slice(dot);
  try {
    const r = await api.rename({ items: [{ id: item.id, v: item.v }], name });
    if (r.done > 0) {
      toast(t('renamed'));
      void store.rescan();
      return true;
    }
    toast(r.error || t('error'), 3500);
  } catch (e) {
    fail(e);
  }
  return false;
}

export function hideFolder(path: string): void {
  if (!path) return;
  store.update({ hidden: [...new Set([...store.settings.hidden, path])] });
}

// --------------------------------------------------------------------- выбор папки

const BASES = ['Pictures/', 'DCIM/', 'Movies/', 'Download/'];

function cleanSegment(s: string): string {
  return s
    .replace(/[\\:*?"<>|]/g, '')
    .replace(/^\.+/, '')
    .trim();
}

export function pickFolder(current?: string): Promise<string | null> {
  return new Promise((resolve) => {
    let picked: string | null = null;
    const list = h('div', { class: 'picker' });

    const choose = (path: string): void => {
      picked = path;
      sh.close();
    };

    const newFolder = async (): Promise<void> => {
      const base = await new Promise<string | null>((res) => {
        menu(
          t('pickFolderIn'),
          BASES.map((b) => ({ icon: 'folder', label: b, onClick: () => res(b) })),
          () => setTimeout(() => res(null), 0)
        );
      });
      if (!base) return;
      const name = await promptDialog(t('newFolder'), '', t('folderName'), t('create'));
      if (name === null) return;
      const clean = name
        .split('/')
        .map(cleanSegment)
        .filter(Boolean)
        .join('/');
      if (clean) choose(`${base}${clean}/`);
    };

    list.append(
      h(
        'button',
        { class: 'menu-item accent', onClick: () => void newFolder() },
        icon('plus', 22),
        h('span', { class: 'menu-label', text: t('newFolder') })
      )
    );
    const folders = [...store.folders.values()]
      .filter((f) => f.direct.length > 0 && f.path !== current)
      .sort((a, b) => a.path.localeCompare(b.path));
    for (const f of folders) {
      list.append(
        h(
          'button',
          { class: 'menu-item', onClick: () => choose(f.path) },
          icon('folder', 22),
          h('span', { class: 'menu-label' }, f.path.replace(/\/$/, ''), h('small', { text: fmtCount(f.direct.length) }))
        )
      );
    }
    const sh = sheet(t('pickFolder'), list, {
      full: true,
      onClose: () => setTimeout(() => resolve(picked), 0),
    });
  });
}

// --------------------------------------------------------------------- информация

export async function showInfo(item: Item): Promise<void> {
  const body = h('div', { class: 'info' });
  const sh = sheet(t('info'), body);
  const row = (k: string, v: string | undefined): void => {
    if (v) body.append(h('div', { class: 'info-row' }, h('span', { class: 'k', text: k }), h('span', { class: 'v', text: v })));
  };
  const base = (): void => {
    body.textContent = '';
    row(t('infoName'), item.name);
    row(t('infoPath'), item.dir);
    row(t('infoSize'), fmtSize(item.size));
    if (item.w && item.h) row(t('infoRes'), `${item.w} × ${item.h} (${((item.w * item.h) / 1e6).toFixed(1)} MP)`);
    row(t('infoDate'), fmtDateTime(item.taken));
    row(t('infoType'), item.mime);
    if (item.v) row(t('infoDuration'), fmtDuration(item.dur));
  };
  base();
  try {
    const i = await api.getInfo({ id: item.id, v: item.v });
    base();
    if (i.path) row(t('infoPath'), i.path);
    row(t('infoCamera'), [i.make, i.model].filter(Boolean).join(' '));
    const exp = [i.exposure ? `${i.exposure} s` : '', i.fnumber ? `ƒ/${i.fnumber}` : '', i.focal ? `${i.focal} mm` : '', i.iso ? `ISO ${i.iso}` : '']
      .filter(Boolean)
      .join(' · ');
    row(t('infoExposure'), exp);
    if (typeof i.lat === 'number' && typeof i.lon === 'number') row(t('infoGps'), `${i.lat.toFixed(5)}, ${i.lon.toFixed(5)}`);
  } catch {
    /* оставляем базовую информацию */
  }
  void sh;
}

export function forgetThumbs(items: Item[]): void {
  thumbs.forget(items.map((i) => i.k));
}
