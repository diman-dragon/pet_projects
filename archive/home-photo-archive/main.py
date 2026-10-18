#!/usr/bin/env python3
"""
MediaOrganizer — Сортировщик и дедупликатор медиафайлов
"""

import os
import re
import sys
import shutil
import hashlib
import logging
import threading
from datetime import datetime
from pathlib import Path
from tkinter import *
from tkinter import ttk, filedialog, messagebox

# ──────────────────────────────────────────────────────────────────────────────
# ОПЦИОНАЛЬНЫЕ ЗАВИСИМОСТИ
# ──────────────────────────────────────────────────────────────────────────────
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PILLOW = True
except ImportError:
    PILLOW = False

try:
    from hachoir.parser import createParser
    from hachoir.metadata import extractMetadata
    import hachoir.core.config as _hcfg
    _hcfg.quiet = True
    HACHOIR = True
except ImportError:
    HACHOIR = False

# ──────────────────────────────────────────────────────────────────────────────
# РАСШИРЕНИЯ ФАЙЛОВ
# ──────────────────────────────────────────────────────────────────────────────
VIDEO_EXT = frozenset({
    '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.m4v', '.3gp',
    '.flv', '.webm', '.mts', '.m2ts', '.ts', '.vob', '.mpg',
    '.mpeg', '.divx', '.rmvb', '.asf', '.ogv', '.f4v', '.rm',
})

ALWAYS_PHOTO_EXT = frozenset({
    '.heic', '.heif', '.raw', '.cr2', '.cr3', '.nef', '.arw',
    '.dng', '.orf', '.rw2', '.pef', '.srw', '.x3f', '.3fr',
    '.raf', '.rwl', '.iiq',
})

MAYBE_PHOTO_EXT = frozenset({
    '.jpg', '.jpeg', '.tiff', '.tif', '.png', '.bmp', '.webp',
})

IMAGE_EXT = frozenset({
    '.png', '.bmp', '.webp', '.ico', '.psd', '.ai', '.eps',
    '.svg', '.xcf', '.pcx', '.tga',
})

GIF_EXT = frozenset({'.gif', '.apng'})

MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December',
}


# ──────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ──────────────────────────────────────────────────────────────────────────────

def compute_hash(path: Path, quick: bool = True) -> str:
    h = hashlib.md5()
    size = path.stat().st_size
    with open(path, 'rb') as f:
        if quick and size > 2 * 1024 * 1024:
            h.update(f.read(131072))
            h.update(size.to_bytes(8, 'little'))
            f.seek(max(0, size - 131072))
            h.update(f.read(131072))
        else:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
    return h.hexdigest()


def get_exif_datetime(path: Path) -> "datetime | None":
    if not PILLOW:
        return None
    try:
        with Image.open(path) as img:
            exif_raw = img._getexif()
            if not exif_raw:
                return None
            priority = ('DateTimeOriginal', 'DateTimeDigitized', 'DateTime')
            found = {}
            for tag_id, val in exif_raw.items():
                name = TAGS.get(tag_id, '')
                if name in priority:
                    found[name] = val
            for field in priority:
                if field in found:
                    try:
                        return datetime.strptime(str(found[field])[:19], '%Y:%m:%d %H:%M:%S')
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return None


def file_has_exif(path: Path) -> bool:
    if not PILLOW:
        return False
    try:
        with Image.open(path) as img:
            exif = img._getexif()
            return bool(exif and len(exif) > 2)
    except Exception:
        return False


def get_video_datetime(path: Path) -> "datetime | None":
    if not HACHOIR:
        return None
    try:
        parser = createParser(str(path))
        if not parser:
            return None
        with parser:
            meta = extractMetadata(parser)
            if not meta:
                return None
            for field in ('creation_date', 'date_time_original', 'last_modification'):
                try:
                    val = meta.get(field)
                    if val is None:
                        continue
                    if isinstance(val, datetime):
                        return val.replace(tzinfo=None)
                    s = str(val).strip()
                    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                                '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d'):
                        try:
                            return datetime.strptime(s[:19], fmt)
                        except ValueError:
                            continue
                except Exception:
                    continue
    except Exception:
        pass
    return None


def get_best_date(path: Path, category: str) -> datetime:
    dt = None
    if category == 'video':
        dt = get_video_datetime(path)
    elif category == 'photo':
        dt = get_exif_datetime(path)

    if dt:
        return dt

    stat = path.stat()
    ts = min(stat.st_mtime, stat.st_ctime)
    return datetime.fromtimestamp(ts)


def get_file_date(path: Path) -> datetime:
    ext = path.suffix.lower()
    if ext in VIDEO_EXT:
        dt = get_video_datetime(path)
    elif ext in ALWAYS_PHOTO_EXT or ext in MAYBE_PHOTO_EXT:
        dt = get_exif_datetime(path)
    else:
        dt = None
    if dt:
        return dt
    stat = path.stat()
    ts = min(stat.st_mtime, stat.st_ctime)
    return datetime.fromtimestamp(ts)


def classify_file(path: Path) -> "str | None":
    ext = path.suffix.lower()
    if ext in GIF_EXT:
        return 'gif'
    if ext in VIDEO_EXT:
        return 'video'
    if ext in ALWAYS_PHOTO_EXT:
        return 'photo'
    if ext in MAYBE_PHOTO_EXT:
        return 'photo' if file_has_exif(path) else 'image'
    if ext in IMAGE_EXT:
        return 'image'
    return None


# ──────────────────────────────────────────────────────────────────────────────
# ОСНОВНОЙ КЛАСС ОБРАБОТКИ
# ──────────────────────────────────────────────────────────────────────────────

class MediaOrganizer:
    CAT_DIRS = {
        'video': 'Видео',
        'photo': 'Фото',
        'image': 'Картинки',
        'gif':   'Гифки',
    }

    def __init__(self, src: str, dst: str, callbacks: dict = None):
        self.src = Path(src).resolve()
        self.dst = Path(dst).resolve()
        self.cb = callbacks or {}
        self._stop_flag = False

        self.dst.mkdir(parents=True, exist_ok=True)
        log_path = self.dst / 'media_organizer.log'

        self.logger = logging.getLogger(f'MediaOrg_{id(self)}')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fh = logging.FileHandler(log_path, encoding='utf-8', mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(fh)

    def stop(self):
        self._stop_flag = True

    def run(self) -> dict:
        self._log('=' * 60)
        self._log('НАЧАЛО ОБРАБОТКИ')
        self._log(f'Источник:   {self.src}')
        self._log(f'Назначение: {self.dst}')
        self._log(f'Pillow:     {"✓" if PILLOW else "✗"}')
        self._log(f'hachoir:    {"✓" if HACHOIR else "✗"}')
        self._log('=' * 60)

        stats = {k: 0 for k in ('video', 'photo', 'image', 'gif', 'duplicates', 'errors', 'skipped', 'total')}

        cat_dirs = {}
        for cat, dirname in self.CAT_DIRS.items():
            d = self.dst / dirname
            d.mkdir(parents=True, exist_ok=True)
            cat_dirs[cat] = d

        # Индексация существующих файлов
        self._set_phase('index')
        self._set_status('Индексация папки назначения...')
        hash_indexes = {cat: {} for cat in self.CAT_DIRS}

        all_existing = []
        for cat, d in cat_dirs.items():
            if d.exists():
                all_existing.extend((cat, f) for f in d.rglob('*') if f.is_file())

        total_idx = len(all_existing)
        for i, (cat, f) in enumerate(all_existing):
            if self._stop_flag:
                break
            try:
                hash_indexes[cat][compute_hash(f)] = f
            except Exception as e:
                self._log(f'Ошибка индексации {f.name}: {e}', 'warning')
            self._cb_progress(i + 1, max(total_idx, 1), f.name)

        # Сканирование источника
        self._set_status('Сканирование источника...')
        all_files = self._collect_files()
        total = len(all_files)
        stats['total'] = total
        self._log(f'Найдено файлов в источнике: {total}')

        # Перенос
        self._set_phase('transfer')
        self._set_status(f'Перенос файлов... (0 / {total})')
        self._cb_progress(0, max(total, 1), '')

        counters: dict = {}

        for i, src_file in enumerate(all_files):
            if self._stop_flag:
                self._log('Обработка остановлена пользователем.', 'warning')
                break

            self._cb_progress(i + 1, total, src_file.name)

            try:
                cat = classify_file(src_file)
                if cat is None:
                    stats['skipped'] += 1
                    continue

                dest_dir = cat_dirs[cat]
                src_hash = compute_hash(src_file)

                if src_hash in hash_indexes[cat]:
                    dup = hash_indexes[cat][src_hash]
                    self._log(f'ДУБЛИКАТ: {src_file.name}  ≡  {dup.name} → удалён')
                    src_file.unlink()
                    stats['duplicates'] += 1
                    continue

                date = get_best_date(src_file, cat)
                ext = src_file.suffix.lower()
                new_name = self._unique_name(dest_dir, date, ext, counters)
                dst_path = dest_dir / new_name

                shutil.move(str(src_file), str(dst_path))
                hash_indexes[cat][src_hash] = dst_path

                self._log(f'[{cat.upper():<5}] {src_file.name:<40} → {new_name}   ({date.strftime("%Y-%m-%d")})')
                stats[cat] += 1

            except Exception as e:
                self._log(f'ОШИБКА {src_file.name}: {e}', 'error')
                stats['errors'] += 1

        # Итог
        lines = [
            '=' * 60, 'ИТОГ', '-' * 60,
            f"Всего файлов:         {stats['total']}",
            f"Видео:                {stats['video']}",
            f"Фото:                 {stats['photo']}",
            f"Картинки:             {stats['image']}",
            f"Гифки:                {stats['gif']}",
            f"Дубликатов удалено:   {stats['duplicates']}",
            f"Ошибок:               {stats['errors']}",
            f"Пропущено:            {stats['skipped']}",
            '=' * 60,
        ]
        for line in lines:
            self._log(line)

        self._cb_done(stats)
        return stats

    def _collect_files(self) -> list:
        result = []
        src_res = self.src.resolve()
        dst_res = self.dst.resolve()
        for f in self.src.rglob('*'):
            if f.is_file():
                try:
                    f.resolve().relative_to(dst_res)
                    continue
                except ValueError:
                    result.append(f)
        return result

    def _unique_name(self, folder: Path, date: datetime, ext: str, counters: dict) -> str:
        date_str = date.strftime('%y%m%d')
        key = (str(folder), date_str, ext)

        if key not in counters:
            max_nn = 0
            pat = re.compile(rf'^{date_str}_(\d+){re.escape(ext)}$', re.IGNORECASE)
            for existing in folder.iterdir():
                m = pat.match(existing.name)
                if m:
                    max_nn = max(max_nn, int(m.group(1)))
            counters[key] = max_nn

        counters[key] += 1
        return f'{date_str}_{counters[key]:02d}{ext}'

    def _log(self, msg: str, level: str = 'info'):
        getattr(self.logger, level)(msg)
        if 'log' in self.cb:
            self.cb['log'](level.upper(), msg)

    def _set_status(self, text: str):
        if 'status' in self.cb:
            self.cb['status'](text)

    def _set_phase(self, phase: str):
        if 'phase' in self.cb:
            self.cb['phase'](phase)

    def _cb_progress(self, current: int, total: int, name: str):
        if 'progress' in self.cb:
            self.cb['progress'](current, total, name)

    def _cb_done(self, stats: dict):
        if 'done' in self.cb:
            self.cb['done'](stats)


# ──────────────────────────────────────────────────────────────────────────────
# СОРТИРОВЩИК ПО ДАТАМ
# ──────────────────────────────────────────────────────────────────────────────

class DateSorter:
    CAT_DIRS = ('Видео', 'Фото', 'Картинки', 'Гифки')

    def __init__(self, dst: str, callbacks: dict = None):
        self.dst = Path(dst).resolve()
        self.cb = callbacks or {}
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def scan(self) -> int:
        total = 0
        for cat_dir in self.CAT_DIRS:
            d = self.dst / cat_dir
            if d.exists():
                total += sum(1 for f in d.iterdir() if f.is_file())
        return total

    def run(self) -> dict:
        stats = {'moved': 0, 'errors': 0, 'total': 0}
        all_files = []

        for cat_name in self.CAT_DIRS:
            cat_root = self.dst / cat_name
            if cat_root.exists():
                all_files.extend((f, cat_root) for f in cat_root.iterdir() if f.is_file())

        total = len(all_files)
        stats['total'] = total

        self._cb_progress(0, max(total, 1), '')
        self._set_status(f'Сортировка по датам... (0 / {total})')

        for i, (src_file, cat_root) in enumerate(all_files):
            if self._stop_flag:
                break
            self._cb_progress(i + 1, total, src_file.name)

            try:
                dt = get_file_date(src_file)
                folder_name = f'{dt.year} - {MONTH_NAMES[dt.month]}'
                dest_dir = cat_root / folder_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                dst_path = dest_dir / src_file.name
                if dst_path.exists():
                    stem, ext = src_file.stem, src_file.suffix
                    n = 1
                    while dst_path.exists():
                        dst_path = dest_dir / f'{stem}_{n}{ext}'
                        n += 1

                shutil.move(str(src_file), str(dst_path))
                stats['moved'] += 1

                if 'log' in self.cb:
                    self.cb['log']('INFO', f'[SORT] {src_file.name} → {cat_root.name}/{folder_name}/')

            except Exception as e:
                stats['errors'] += 1
                if 'log' in self.cb:
                    self.cb['log']('ERROR', f'[SORT] ОШИБКА {src_file.name}: {e}')

        self._cb_done(stats)
        return stats

    def _set_status(self, text: str):
        if 'status' in self.cb:
            self.cb['status'](text)

    def _cb_progress(self, current: int, total: int, name: str):
        if 'progress' in self.cb:
            self.cb['progress'](current, total, name)

    def _cb_done(self, stats: dict):
        if 'done' in self.cb:
            self.cb['done'](stats)


# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────

C = {
    'BG': '#1e1e2e', 'PANEL': '#2a2a3e', 'SURFACE': '#313244',
    'BORDER': '#45475a', 'TEXT': '#cdd6f4', 'SUBTEXT': '#a6adc8',
    'MUTED': '#6c7086', 'ACCENT': '#7c6af7', 'GREEN': '#a6e3a1',
    'RED': '#f38ba8', 'YELLOW': '#f9e2af', 'BLUE': '#89b4fa',
    'CYAN': '#94e2d5', 'MAUVE': '#cba6f7', 'ORANGE': '#fab387',
}


class ProgressBlock(Frame):
    def __init__(self, parent, label: str, color: str, **kwargs):
        super().__init__(parent, bg=C['PANEL'], **kwargs)
        self._color = color

        # Фикс стиля Progressbar
        style = ttk.Style()
        style_name = f"Custom_{id(self)}.Horizontal.TProgressbar"

        style.configure(style_name,
                        background=color,
                        troughcolor=C['SURFACE'],
                        borderwidth=0,
                        thickness=8,
                        lightcolor=color,
                        darkcolor=color)

        # Явно задаём layout (важно для Windows)
        style.layout(style_name, [
            ('Horizontal.Progressbar.trough', {
                'children': [('Horizontal.Progressbar.pbar', {'side': 'left', 'sticky': 'ns'})]
            })
        ])

        top = Frame(self, bg=C['PANEL'])
        top.pack(fill=X)

        self._phase_lbl = Label(top, text=label, bg=C['PANEL'], fg=C['SUBTEXT'],
                                font=('Segoe UI', 9))
        self._phase_lbl.pack(side=LEFT)

        self._pct_lbl = Label(top, text='', bg=C['PANEL'], fg=color,
                              font=('Segoe UI', 9, 'bold'))
        self._pct_lbl.pack(side=RIGHT)

        self._bar = ttk.Progressbar(self, mode='determinate', style=style_name)
        self._bar.pack(fill=X, pady=(3, 2))

        self._file_lbl = Label(self, text='', bg=C['PANEL'], fg=C['MUTED'],
                               font=('Consolas', 8), anchor=W)
        self._file_lbl.pack(fill=X)

    def update_progress(self, current: int, total: int, name: str = ''):
        self._bar['maximum'] = total
        self._bar['value'] = current
        pct = int(current / total * 100) if total > 0 else 0
        self._pct_lbl.configure(text=f'{pct}%')
        disp = name if len(name) <= 70 else '…' + name[-67:]
        self._file_lbl.configure(text=disp)

    def reset(self, label: str = None):
        if label:
            self._phase_lbl.configure(text=label)
        self._bar['value'] = 0
        self._pct_lbl.configure(text='')
        self._file_lbl.configure(text='')

    def set_done(self):
        self._bar['value'] = self._bar['maximum']
        self._pct_lbl.configure(text='100%')

    def set_active(self, active: bool):
        fg = self._color if active else C['MUTED']
        self._phase_lbl.configure(fg=fg)


class App(Tk):
    def __init__(self):
        super().__init__()
        self.title("MediaOrganizer — Сортировщик медиафайлов")
        self.geometry("880x740")
        self.minsize(720, 600)
        self.configure(bg=C['BG'])

        self._organizer = None
        self._sorter = None
        self._thread = None
        self._current_phase = 'index'

        self._build_ui()
        self._check_deps()

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        root = ttk.Frame(self, padding=(24, 20, 24, 16))
        root.pack(fill=BOTH, expand=True)

        # Заголовок
        hdr = ttk.Frame(root)
        hdr.pack(fill=X, pady=(0, 14))
        ttk.Label(hdr, text='🗂  MediaOrganizer', style='Head.TLabel').pack(side=LEFT)
        self._dep_label = ttk.Label(hdr, text='', style='Sub.TLabel')
        self._dep_label.pack(side=RIGHT, anchor=E)

        # Выбор папок
        pf = Frame(root, bg=C['PANEL'], bd=0, highlightthickness=1,
                   highlightbackground=C['BORDER'])
        pf.pack(fill=X, pady=(0, 12))
        inner = Frame(pf, bg=C['PANEL'])
        inner.pack(fill=X, padx=16, pady=14)
        inner.columnconfigure(1, weight=1)

        self._src_var = StringVar()
        self._dst_var = StringVar()

        for text, var, cmd, row in [
            ('📂  Папка источника:', self._src_var, self._choose_src, 0),
            ('📦  Папка назначения:', self._dst_var, self._choose_dst, 1)
        ]:
            Label(inner, text=text, bg=C['PANEL'], fg=C['SUBTEXT'],
                  font=('Segoe UI', 10)).grid(row=row, column=0, sticky=W, padx=(0,10), pady=5)
            Entry(inner, textvariable=var, bg=C['SURFACE'], fg=C['TEXT'],
                  insertbackground=C['TEXT'], relief=FLAT, font=('Segoe UI', 10),
                  highlightthickness=1, highlightcolor=C['ACCENT'],
                  highlightbackground=C['BORDER']).grid(row=row, column=1, sticky=EW, pady=5)
            Button(inner, text='Обзор', command=cmd, bg=C['ACCENT'], fg='white',
                   font=('Segoe UI', 9, 'bold'), relief=FLAT, cursor='hand2',
                   padx=14, pady=5).grid(row=row, column=2, padx=(10,0), pady=5)

        # Кнопки
        ctrl = Frame(root, bg=C['BG'])
        ctrl.pack(fill=X, pady=(0, 12))

        self._start_btn = Button(ctrl, text='▶   Начать обработку',
                                 command=self._on_start, bg=C['GREEN'], fg='#1e1e2e',
                                 font=('Segoe UI', 11, 'bold'), relief=FLAT,
                                 cursor='hand2', padx=22, pady=9)
        self._start_btn.pack(side=LEFT)

        self._stop_btn = Button(ctrl, text='⏹   Стоп', command=self._on_stop,
                                bg=C['RED'], fg='#1e1e2e', font=('Segoe UI', 11, 'bold'),
                                relief=FLAT, cursor='hand2', padx=22, pady=9, state=DISABLED)
        self._stop_btn.pack(side=LEFT, padx=(10, 0))

        Frame(ctrl, bg=C['BORDER'], width=2).pack(side=LEFT, fill=Y, padx=(16, 16))

        self._sort_btn = Button(ctrl, text='📅   Сортировать по датам',
                                command=self._on_sort, bg=C['ORANGE'], fg='#1e1e2e',
                                font=('Segoe UI', 11, 'bold'), relief=FLAT,
                                cursor='hand2', padx=22, pady=9)
        self._sort_btn.pack(side=LEFT)

        # Статус и прогресс
        self._status_var = StringVar(value='Ожидание...')
        Label(root, textvariable=self._status_var, bg=C['BG'], fg=C['TEXT'],
              font=('Segoe UI', 10, 'bold')).pack(anchor=W, pady=(0,8))

        pgf = Frame(root, bg=C['PANEL'], highlightthickness=1, highlightbackground=C['BORDER'])
        pgf.pack(fill=X, pady=(0, 12))
        pgi = Frame(pgf, bg=C['PANEL'])
        pgi.pack(fill=X, padx=16, pady=12)

        self._pb_index = ProgressBlock(pgi, '📋  Индексация существующих файлов', C['BLUE'])
        self._pb_index.pack(fill=X, pady=(0, 8))

        Frame(pgi, bg=C['BORDER'], height=1).pack(fill=X, pady=4)

        self._pb_transfer = ProgressBlock(pgi, '🚚  Перенос файлов', C['ACCENT'])
        self._pb_transfer.pack(fill=X, pady=(8, 0))

        # Лог
        log_hdr = ttk.Frame(root)
        log_hdr.pack(fill=X)
        ttk.Label(log_hdr, text='Журнал:', style='Sub.TLabel').pack(side=LEFT)
        Button(log_hdr, text='Очистить', command=self._clear_log,
               bg=C['SURFACE'], fg=C['MUTED'], font=('Segoe UI', 8),
               relief=FLAT, cursor='hand2', padx=8, pady=2).pack(side=RIGHT)

        log_wrap = Frame(root, bg=C['PANEL'], highlightthickness=1,
                         highlightbackground=C['BORDER'])
        log_wrap.pack(fill=BOTH, expand=True, pady=(4, 0))

        self._log_text = Text(log_wrap, bg='#11111b', fg=C['TEXT'],
                              font=('Consolas', 9), relief=FLAT, wrap=WORD,
                              padx=10, pady=8, state=DISABLED)
        sb = ttk.Scrollbar(log_wrap, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=sb.set)
        sb.pack(side=RIGHT, fill=Y)
        self._log_text.pack(fill=BOTH, expand=True)

        for tag, color in [('INFO', C['SUBTEXT']), ('WARNING', C['YELLOW']),
                           ('ERROR', C['RED']), ('VIDEO', C['BLUE']),
                           ('PHOTO', C['GREEN']), ('IMAGE', C['CYAN']),
                           ('GIF', C['MAUVE']), ('DUP', C['YELLOW']),
                           ('SORT', C['ORANGE']), ('SEP', C['BORDER'])]:
            self._log_text.tag_config(tag, foreground=color)

    def _check_deps(self):
        parts = []
        if not PILLOW: parts.append('⚠ Pillow не установлен')
        if not HACHOIR: parts.append('⚠ hachoir не установлен')
        self._dep_label.configure(
            text='  '.join(parts) if parts else '✓ Все зависимости доступны',
            foreground=C['RED'] if parts else C['GREEN']
        )

    def _choose_src(self):
        d = filedialog.askdirectory(title='Папка источника')
        if d: self._src_var.set(d)

    def _choose_dst(self):
        d = filedialog.askdirectory(title='Папка назначения')
        if d: self._dst_var.set(d)

    def _set_busy(self, busy: bool):
        state = DISABLED if busy else NORMAL
        self._start_btn.configure(state=state)
        self._sort_btn.configure(state=state)
        self._stop_btn.configure(state=NORMAL if busy else DISABLED)

    # Остальные методы (_on_start, _on_stop, _on_sort, callbacks) оставлены без изменений
    # (они уже были правильными)

    def _on_start(self):
        src = self._src_var.get().strip()
        dst = self._dst_var.get().strip()
        if not src or not dst:
            messagebox.showerror('Ошибка', 'Укажите обе папки')
            return
        if not Path(src).exists():
            messagebox.showerror('Ошибка', 'Папка источника не существует')
            return
        if Path(src).resolve() == Path(dst).resolve():
            messagebox.showerror('Ошибка', 'Папки источника и назначения не должны совпадать')
            return

        self._set_busy(True)
        self._status_var.set('Запуск...')
        self._pb_index.reset()
        self._pb_transfer.reset()
        self._pb_index.set_active(True)
        self._pb_transfer.set_active(False)

        self._organizer = MediaOrganizer(
            src=src, dst=dst,
            callbacks={
                'log': self._cb_log,
                'progress': self._cb_progress,
                'status': self._cb_status,
                'phase': self._cb_phase,
                'done': self._cb_done,
            }
        )
        self._thread = threading.Thread(target=self._organizer.run, daemon=True)
        self._thread.start()

    def _on_stop(self):
        if self._organizer: self._organizer.stop()
        if self._sorter: self._sorter.stop()
        self._stop_btn.configure(state=DISABLED)
        self._status_var.set('Остановка...')

    def _on_sort(self):
        dst = self._dst_var.get().strip()
        if not dst or not Path(dst).exists():
            messagebox.showerror('Ошибка', 'Укажите корректную папку назначения')
            return

        sorter_check = DateSorter(dst)
        n = sorter_check.scan()
        if n == 0:
            messagebox.showinfo('Инфо', 'Нет файлов для сортировки')
            return

        if not messagebox.askyesno('Подтверждение', f'Сортировать {n} файлов по датам?'):
            return

        self._set_busy(True)
        self._sorter = DateSorter(dst, callbacks={
            'log': self._cb_log,
            'progress': self._cb_sort_progress,
            'status': self._cb_status,
            'done': self._cb_sort_done,
        })
        self._thread = threading.Thread(target=self._sorter.run, daemon=True)
        self._thread.start()

    def _cb_phase(self, phase: str):
        def _do():
            if phase == 'index':
                self._pb_index.set_active(True)
                self._pb_transfer.set_active(False)
            elif phase == 'transfer':
                self._pb_index.set_done()
                self._pb_index.set_active(False)
                self._pb_transfer.set_active(True)
            self._current_phase = phase
        self.after(0, _do)

    def _cb_progress(self, current: int, total: int, name: str):
        def _do():
            if getattr(self, '_current_phase', 'index') == 'index':
                self._pb_index.update_progress(current, total, name)
            else:
                self._pb_transfer.update_progress(current, total, name)
            self._status_var.set(f'Обработка: {current}/{total}')
        self.after(0, _do)

    def _cb_sort_progress(self, current: int, total: int, name: str):
        def _do():
            self._pb_transfer.update_progress(current, total, name)
            self._status_var.set(f'Сортировка: {current}/{total}')
        self.after(0, _do)

    def _cb_status(self, text: str):
        self.after(0, lambda: self._status_var.set(text))

    def _cb_log(self, level: str, msg: str):
        def _do():
            self._log_text.configure(state=NORMAL)
            ts = datetime.now().strftime('%H:%M:%S')

            if '=' in msg or '-'*10 in msg:
                tag = 'SEP'
            elif 'ДУБЛИКАТ' in msg:
                tag = 'DUP'
            elif '[SORT]' in msg:
                tag = 'SORT'
            elif level == 'ERROR':
                tag = 'ERROR'
            elif level == 'WARNING':
                tag = 'WARNING'
            else:
                tag = level if level in ('VIDEO','PHOTO','IMAGE','GIF') else 'INFO'

            self._log_text.insert(END, f'[{ts}] {msg}\n', tag)
            self._log_text.see(END)
            self._log_text.configure(state=DISABLED)
        self.after(0, _do)

    def _cb_done(self, stats: dict):
        def _do():
            self._set_busy(False)
            self._pb_index.set_done()
            self._pb_transfer.set_done()
            self._status_var.set('✅ Обработка завершена')
            messagebox.showinfo('Готово', f'Обработка завершена!\n\n'
                                          f'Видео: {stats["video"]}\n'
                                          f'Фото: {stats["photo"]}\n'
                                          f'Картинки: {stats["image"]}\n'
                                          f'Гифки: {stats["gif"]}\n'
                                          f'Дубликаты: {stats["duplicates"]}')
        self.after(0, _do)

    def _cb_sort_done(self, stats: dict):
        def _do():
            self._set_busy(False)
            self._pb_transfer.set_done()
            self._status_var.set('✅ Сортировка завершена')
            messagebox.showinfo('Готово', f'Перемещено файлов: {stats["moved"]}\nОшибок: {stats["errors"]}')
        self.after(0, _do)

    def _clear_log(self):
        self._log_text.configure(state=NORMAL)
        self._log_text.delete('1.0', END)
        self._log_text.configure(state=DISABLED)


if __name__ == '__main__':
    app = App()
    app.mainloop()