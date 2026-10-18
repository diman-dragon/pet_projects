#!/usr/bin/env python3
"""
DataLaw Camera — генератор брендовых ассетов из логотипа.

Вход : brand/source/logo-original.png  (чёрный глянцевый «D» на белом фоне)
Выход: android-res/**           (иконки, сплэш, водяной знак — копируются в android/ скриптом patch-android.mjs)
       www/brand/**             (знаки для интерфейса)
       brand/out/**             (иконка Google Play 512, feature graphic 1024x500, preview.png)

Запуск:  pip install pillow numpy scipy  &&  python3 brand/build_assets.py
Меняете логотип или цвета — запускаете заново и коммитите результат.
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "brand" / "source" / "logo-original.png"
OUT_RES = ROOT / "android-res"
OUT_WWW = ROOT / "www" / "brand"
OUT_STORE = ROOT / "brand" / "out"

ACCENT = (76, 141, 255)       # DataLaw Blue
INK = (11, 12, 16)
PORCELAIN_TOP = (251, 252, 254)
PORCELAIN_BOTTOM = (212, 217, 226)
SPLASH_BG = (244, 246, 249)

# Положение линзы в координатах исходного изображения (между стволом «D» и полумесяцем)
LENS_CX, LENS_CY, LENS_R = 909.0, 540.0, 122.0


def mkdir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def save(img: Image.Image, path: Path):
    mkdir(path)
    img.save(path, optimize=True)
    print("  +", path.relative_to(ROOT))


# --------------------------------------------------------------------------- логотип
def load_logo():
    """Возвращает RGBA-логотип (без белого фона) и bbox/центры в пикселях исходника."""
    im = Image.open(SRC).convert("RGB")
    a = np.asarray(im).astype(np.float32)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    mask = lum < 128                                   # тело логотипа (тень светлее 190)
    mask = ndi.binary_fill_holes(mask)
    soft = ndi.gaussian_filter(mask.astype(np.float32), 0.8)  # антиалиасинг края
    # цвет краевых пикселей берём у ближайших «внутренних» — без белой каймы
    solid = soft > 0.98
    idx = ndi.distance_transform_edt(~solid, return_distances=False, return_indices=True)
    rgb = a[idx[0], idx[1]]
    rgba = np.dstack([rgb, soft * 255.0]).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    centroid = (xs.mean(), ys.mean())
    return Image.fromarray(rgba, "RGBA"), bbox, centroid, mask


# --------------------------------------------------------------------------- линза
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def over(dst, rgb, alpha):
    """alpha-композит поверх dst (float RGBA 0..1); rgb — (...,3) 0..1; alpha — (...)"""
    a = np.clip(alpha, 0, 1)[..., None]
    da = dst[..., 3:4]
    out_a = a + da * (1 - a)
    out_rgb = (rgb * a + dst[..., :3] * da * (1 - a)) / np.maximum(out_a, 1e-6)
    dst[..., :3] = out_rgb
    dst[..., 3:4] = out_a


def render_lens(size: int, accent=ACCENT) -> Image.Image:
    """Глянцевый объектив: оправа, кольцо с насечкой, стекло с просветлением и бликами."""
    ss = 3
    n = size * ss
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2.0
    dx, dy = (xx - c) / (n / 2.0), (yy - c) / (n / 2.0)
    r = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    acc = np.array(accent, np.float32) / 255.0
    out = np.zeros((n, n, 4), np.float32)

    # 1) оправа (металл, свет сверху-слева)
    t = (dy + 1) / 2
    bezel = (np.array([0.40, 0.42, 0.47]) * (1 - t[..., None]) + np.array([0.05, 0.05, 0.07]) * t[..., None])
    bezel_mask = (r <= 1.0) * (r >= 0.80)
    over(out, bezel, bezel_mask.astype(np.float32) * smoothstep(1.0, 0.985, r))
    rim_light = np.clip(np.cos(ang + 2.36), 0, 1) ** 2            # блик на внешней кромке
    over(out, np.ones((n, n, 3), np.float32), 0.55 * rim_light * smoothstep(0.93, 0.985, r) * (r <= 1.0))

    # 2) внутреннее кольцо с насечкой
    ridge = 0.5 + 0.5 * np.cos(r * 2 * np.pi * 34)
    ring_col = (0.06 + 0.07 * ridge)[..., None] * np.ones(3, np.float32)
    over(out, ring_col, ((r < 0.80) * (r >= 0.70)).astype(np.float32))
    over(out, np.ones((n, n, 3), np.float32), 0.22 * ((r > 0.795) * (r < 0.81)))

    # 3) стекло: глубокий синий -> чёрный
    glass_t = np.clip(r / 0.70, 0, 1)
    glass = (np.array([0.03, 0.07, 0.17]) * (1 - glass_t[..., None] ** 1.4)
             + np.array([0.015, 0.016, 0.03]) * glass_t[..., None] ** 1.4)
    glass_mask = (r < 0.70).astype(np.float32)
    over(out, glass, glass_mask)

    # 4) просветляющее покрытие: радужное кольцо + акцентный сектор
    hue = (ang / (2 * np.pi) + 0.62) % 1.0
    h6 = (hue * 6.0)
    k = np.floor(h6)
    f = h6 - k
    p_, q_, t_ = 0.85 * (1 - 0.75), 0.85 * (1 - 0.75 * f), 0.85 * (1 - 0.75 * (1 - f))
    v_ = 0.85
    ki = (k.astype(int)) % 6
    R = np.choose(ki, [v_, q_, p_, p_, t_, v_])
    G = np.choose(ki, [t_, v_, v_, q_, p_, p_])
    B = np.choose(ki, [p_, p_, t_, v_, v_, q_])
    irid = np.stack([R, G, B], -1)
    band = smoothstep(0.38, 0.60, r) * (1 - smoothstep(0.60, 0.70, r))
    over(out, irid, 0.30 * band * glass_mask)
    sector = np.clip(np.cos(ang - 0.9), 0, 1) ** 2
    over(out, np.broadcast_to(acc, (n, n, 3)), 0.55 * sector * (glass_t ** 2) * glass_mask)

    # 5) диафрагма (7 лепестков) в центре — «это камера»
    blades = 7
    a_mod = (ang * blades / (2 * np.pi)) % 1.0
    line = np.exp(-((a_mod - 0.5) / 0.035) ** 2)
    over(out, np.ones((n, n, 3), np.float32), 0.20 * line * (r < 0.34) * (r > 0.07))
    over(out, np.array([0.0, 0.0, 0.0], np.float32) + 0.0, 0.65 * (r < 0.09) * glass_mask)
    over(out, np.ones((n, n, 3), np.float32), 0.20 * ((r > 0.335) * (r < 0.35)))

    # 6) блики
    def ellipse(cx, cy, rx, ry, rot):
        ca, sa = np.cos(rot), np.sin(rot)
        u = (dx - cx) * ca + (dy - cy) * sa
        v = -(dx - cx) * sa + (dy - cy) * ca
        return np.hypot(u / rx, v / ry)

    d1 = ellipse(-0.27, -0.33, 0.42, 0.19, -0.62)
    over(out, np.ones((n, n, 3), np.float32), 0.50 * np.clip(1 - d1, 0, 1) ** 1.6 * glass_mask)
    d2 = np.hypot(dx - 0.37, dy - 0.30)
    over(out, np.ones((n, n, 3), np.float32), 0.70 * np.clip(1 - d2 / 0.07, 0, 1) ** 1.5 * glass_mask)
    d3 = np.hypot(dx - 0.30, dy - 0.38)
    over(out, np.ones((n, n, 3), np.float32), 0.35 * np.clip(1 - d3 / 0.03, 0, 1) * glass_mask)

    img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGBA")
    return img.resize((size, size), Image.LANCZOS)


def lens_silhouette(size: int) -> Image.Image:
    """Монохромная версия линзы для themed-иконки Android 13+ (только альфа)."""
    ss = 3
    n = size * ss
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2.0
    r = np.hypot((xx - c) / (n / 2.0), (yy - c) / (n / 2.0))
    a = ((r <= 1.0) & (r >= 0.80)) | (r < 0.30) & (r > 0.08) | (r < 0.56) & (r > 0.50)
    alpha = (a * 255).astype(np.uint8)
    img = Image.new("RGBA", (n, n), (255, 255, 255, 0))
    img.putalpha(Image.fromarray(alpha, "L"))
    return img.resize((size, size), Image.LANCZOS)


# --------------------------------------------------------------------------- композиции
class Art:
    """Логотип + линза, готовые к размещению: хранит геометрию в координатах исходника."""

    def __init__(self):
        self.logo, self.bbox, self.centroid, self.mask = load_logo()
        x0, y0, x1, y1 = self.bbox
        bc = ((x0 + x1) / 2, (y0 + y1) / 2)
        # оптический центр: среднее между центром bbox и центром масс
        self.cx = 0.5 * bc[0] + 0.5 * self.centroid[0]
        self.cy = 0.5 * bc[1] + 0.5 * self.centroid[1]
        ys, xs = np.nonzero(self.mask)
        self.max_r = float(np.hypot(xs - self.cx, ys - self.cy).max())
        self._lens_cache = {}

    def layer(self, canvas: int, fill_radius: float = 0.325, shadow: bool = True, shadow_alpha: float = 0.30) -> Image.Image:
        """Передний план на прозрачном холсте canvas×canvas. fill_radius — доля холста до самой дальней точки."""
        s = fill_radius * canvas / self.max_r
        lw, lh = int(round(self.logo.width * s)), int(round(self.logo.height * s))
        logo = self.logo.resize((lw, lh), Image.LANCZOS)
        ox = int(round(canvas / 2 - self.cx * s))
        oy = int(round(canvas / 2 - self.cy * s))

        art = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
        art.alpha_composite(logo, (ox, oy)) if (ox >= 0 and oy >= 0) else art.paste(logo, (ox, oy), logo)

        lr = int(round(LENS_R * s))
        lens = render_lens(lr * 2)
        lx = int(round(ox + LENS_CX * s - lr))
        ly = int(round(oy + LENS_CY * s - lr))
        art.alpha_composite(lens, (lx, ly)) if (lx >= 0 and ly >= 0) else art.paste(lens, (lx, ly), lens)

        if not shadow:
            return art
        alpha = art.getchannel("A")
        sh = Image.new("RGBA", art.size, (8, 12, 28, 0))
        sh.putalpha(alpha.point(lambda v: int(v * shadow_alpha)))
        sh = sh.filter(ImageFilter.GaussianBlur(canvas * 0.012))
        out = Image.new("RGBA", art.size, (0, 0, 0, 0))
        out.alpha_composite(sh, (0, int(canvas * 0.014)))
        out.alpha_composite(art)
        return out

    def mono(self, canvas: int, fill_radius: float = 0.325) -> Image.Image:
        s = fill_radius * canvas / self.max_r
        lw, lh = int(round(self.logo.width * s)), int(round(self.logo.height * s))
        logo = self.logo.resize((lw, lh), Image.LANCZOS)
        ox = int(round(canvas / 2 - self.cx * s))
        oy = int(round(canvas / 2 - self.cy * s))
        a = Image.new("L", (canvas, canvas), 0)
        a.paste(logo.getchannel("A"), (ox, oy))
        lr = int(round(LENS_R * s))
        sil = lens_silhouette(lr * 2).getchannel("A")
        a.paste(Image.new("L", sil.size, 255), (int(round(ox + LENS_CX * s - lr)), int(round(oy + LENS_CY * s - lr))), sil)
        out = Image.new("RGBA", (canvas, canvas), (255, 255, 255, 0))
        out.putalpha(a)
        return out

    def tight(self, height: int, with_lens: bool = True) -> Image.Image:
        """Логотип, обрезанный по bbox, заданной высоты — для UI / водяного знака."""
        x0, y0, x1, y1 = self.bbox
        pad = 8
        crop = self.logo.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
        s = height / crop.height
        crop = crop.resize((int(round(crop.width * s)), height), Image.LANCZOS)
        if with_lens:
            lr = int(round(LENS_R * s))
            lens = render_lens(lr * 2)
            crop.alpha_composite(lens, (int(round((LENS_CX - (x0 - pad)) * s - lr)), int(round((LENS_CY - (y0 - pad)) * s - lr))))
        return crop

    def white_mark(self, height: int) -> Image.Image:
        x0, y0, x1, y1 = self.bbox
        pad = 8
        crop = self.logo.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
        s = height / crop.height
        crop = crop.resize((int(round(crop.width * s)), height), Image.LANCZOS)
        a = crop.getchannel("A")
        out = Image.new("RGBA", crop.size, (255, 255, 255, 0))
        out.putalpha(a)
        # линзу рисуем кольцом в акцентном цвете — узнаваемо и в белом знаке
        lr = LENS_R * s
        d = ImageDraw.Draw(out)
        cx = (LENS_CX - (x0 - pad)) * s
        cy = (LENS_CY - (y0 - pad)) * s
        d.ellipse((cx - lr, cy - lr, cx + lr, cy + lr), outline=ACCENT + (255,), width=max(2, int(lr * 0.18)))
        d.ellipse((cx - lr * 0.35, cy - lr * 0.35, cx + lr * 0.35, cy + lr * 0.35), fill=ACCENT + (255,))
        return out


def background(canvas: int) -> Image.Image:
    """Фарфоровый градиент + едва заметные «круги по воде» вокруг линзы."""
    yy, xx = np.mgrid[0:canvas, 0:canvas].astype(np.float32)
    t = (xx * 0.35 + yy * 0.65) / canvas
    top, bot = np.array(PORCELAIN_TOP, np.float32), np.array(PORCELAIN_BOTTOM, np.float32)
    img = top[None, None, :] * (1 - t[..., None]) + bot[None, None, :] * t[..., None]
    # мягкий световой центр
    glow = np.exp(-(((xx - canvas * 0.42) ** 2 + (yy - canvas * 0.36) ** 2) / (2 * (canvas * 0.34) ** 2)))
    img += glow[..., None] * np.array([6, 6, 6], np.float32)
    # кольца
    cx, cy = canvas * 0.58, canvas * 0.50
    r = np.hypot(xx - cx, yy - cy) / canvas
    rings = np.exp(-(((r * 11) % 1.0 - 0.5) / 0.045) ** 2) * np.clip(1 - r * 1.4, 0, 1)
    img -= rings[..., None] * 10.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGB").convert("RGBA")


def masked(img: Image.Image, shape: str) -> Image.Image:
    n = img.width
    m = Image.new("L", (n * 4, n * 4), 0)
    d = ImageDraw.Draw(m)
    if shape == "circle":
        d.ellipse((0, 0, n * 4 - 1, n * 4 - 1), fill=255)
    else:
        d.rounded_rectangle((0, 0, n * 4 - 1, n * 4 - 1), radius=int(n * 4 * 0.225), fill=255)
    m = m.resize((n, n), Image.LANCZOS)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), m)
    return out


def find_font(bold=True, size=48):
    names = (["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf", "Helvetica.ttc"] if bold
             else ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"])
    dirs = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/Library/Fonts", "/System/Library/Fonts",
            "C:/Windows/Fonts", "/usr/share/fonts/truetype/liberation"]
    for d in dirs:
        for n in names:
            p = Path(d) / n
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def spaced_text(draw, xy, text, font, fill, spacing):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + spacing
    return x


# --------------------------------------------------------------------------- сборка
def main():
    if not SRC.exists():
        sys.exit(f"Нет исходника логотипа: {SRC}")
    print("Логотип:", SRC.relative_to(ROOT))
    art = Art()

    # --- adaptive-слои (108dp * 6 = 648px)
    C = 648
    fg = art.layer(C, fill_radius=0.318)
    bg = background(C)
    mono = art.mono(C, fill_radius=0.318)
    save(bg.convert("RGB"), OUT_RES / "drawable-nodpi" / "dl_ic_background.png")
    save(fg, OUT_RES / "drawable-nodpi" / "dl_ic_foreground.png")
    save(mono, OUT_RES / "drawable-nodpi" / "dl_ic_monochrome.png")

    # --- сплэш (288dp-иконка; видимый круг = 2/3 холста — как у adaptive)
    splash = art.layer(960, fill_radius=0.318, shadow=True, shadow_alpha=0.22)
    save(splash, OUT_RES / "drawable-nodpi" / "dl_splash_icon.png")

    # --- splash.png: сплошной цвет (шаблон Capacitor ссылается на @drawable/splash; растягиваемая картинка исказилась бы)
    save(Image.new("RGB", (64, 64), SPLASH_BG), OUT_RES / "drawable" / "splash.png")

    # --- legacy mipmap (48dp = центральные 2/3 adaptive-композиции)
    comp = bg.copy()
    comp.alpha_composite(fg)
    c0 = int(C * (1 - 72 / 108) / 2)
    center = comp.crop((c0, c0, C - c0, C - c0))
    sizes = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}
    fg_sizes = {"mdpi": 108, "hdpi": 162, "xhdpi": 216, "xxhdpi": 324, "xxxhdpi": 432}
    for dens, px in sizes.items():
        big = center.resize((px * 4, px * 4), Image.LANCZOS)
        save(masked(big, "squircle").resize((px, px), Image.LANCZOS), OUT_RES / f"mipmap-{dens}" / "ic_launcher.png")
        save(masked(big, "circle").resize((px, px), Image.LANCZOS), OUT_RES / f"mipmap-{dens}" / "ic_launcher_round.png")
        save(fg.resize((fg_sizes[dens],) * 2, Image.LANCZOS), OUT_RES / f"mipmap-{dens}" / "ic_launcher_foreground.png")

    # --- знаки для интерфейса
    save(art.white_mark(256), OUT_WWW / "dl_mark_white.png")
    save(art.tight(384, with_lens=True), OUT_WWW / "dl_logo.png")
    save(art.layer(512, fill_radius=0.40, shadow=False), OUT_WWW / "dl_logo_square.png")

    # --- водяной знак: белый «D» + слово DataLaw (для фото)
    mark = art.white_mark(150)
    font = find_font(True, 92)
    tw = ImageDraw.Draw(Image.new("L", (1, 1))).textlength("DataLaw", font=font)
    wm_w = int(mark.width + 28 + tw + 12 + 7 * 3)
    wm = Image.new("RGBA", (wm_w, 190), (0, 0, 0, 0))
    wm.alpha_composite(mark, (6, 20))
    d = ImageDraw.Draw(wm)
    spaced_text(d, (mark.width + 34, 38), "DataLaw", font, (255, 255, 255, 255), 3)
    sh = Image.new("RGBA", wm.size, (0, 0, 0, 0))
    sh.putalpha(wm.getchannel("A").point(lambda v: int(v * 0.55)))
    sh = sh.filter(ImageFilter.GaussianBlur(5))
    final = Image.new("RGBA", wm.size, (0, 0, 0, 0))
    final.alpha_composite(sh, (0, 4))
    final.alpha_composite(wm)
    bbox = final.getbbox()
    final = final.crop(bbox)
    save(final, OUT_RES / "drawable-nodpi" / "dl_watermark.png")
    save(final, OUT_WWW / "dl_watermark.png")

    # --- Google Play: иконка 512
    store = center.resize((1024, 1024), Image.LANCZOS).convert("RGB").resize((512, 512), Image.LANCZOS)
    save(store, OUT_STORE / "play-icon-512.png")

    # --- Google Play: feature graphic 1024x500
    W, H = 1024, 500
    base = background(1100).crop((0, 300, 1100, 800)).resize((W, H), Image.LANCZOS).convert("RGBA")
    logo = art.layer(520, fill_radius=0.43, shadow=True)
    base.alpha_composite(logo, (40, -10))
    d = ImageDraw.Draw(base)
    f1, f2, f3 = find_font(True, 74), find_font(False, 31), find_font(True, 22)
    spaced_text(d, (500, 150), "DataLaw", f1, INK + (255,), 1)
    d.text((502, 238), "Camera", font=find_font(False, 74), fill=(70, 74, 86, 255))
    d.rounded_rectangle((502, 336, 502 + 96, 342), radius=3, fill=ACCENT + (255,))
    d.text((502, 362), "Pro · Portrait · Fun effects", font=f2, fill=(60, 64, 76, 255))
    spaced_text(d, (502, 412), "ALL PROCESSING ON-DEVICE", f3, (110, 116, 130, 255), 3)
    save(base.convert("RGB"), OUT_STORE / "play-feature-graphic-1024x500.png")

    # --- превью-лист для проверки глазами
    sheet = Image.new("RGB", (1600, 760), (30, 32, 38))
    x = 30
    for shape in ("squircle", "circle"):
        big = masked(center.resize((360, 360), Image.LANCZOS), shape)
        sheet.paste(big, (x, 30), big)
        x += 390
    small = masked(center.resize((120, 120), Image.LANCZOS), "squircle")
    sheet.paste(small, (30, 430), small)
    small = masked(center.resize((72, 72), Image.LANCZOS), "circle")
    sheet.paste(small, (170, 460), small)
    # monochrome на тёмном
    m = Image.new("RGBA", (360, 360), (40, 44, 56, 255))
    mm = art.mono(360, fill_radius=0.318)
    tint = Image.new("RGBA", mm.size, (170, 200, 255, 255))
    tint.putalpha(mm.getchannel("A"))
    m.alpha_composite(tint)
    sheet.paste(masked(m, "circle"), (790, 30), masked(m, "circle"))
    # сплэш
    sp = Image.new("RGBA", (200, 360), SPLASH_BG + (255,))
    ic = splash.resize((300, 300), Image.LANCZOS)
    sp.alpha_composite(ic.crop((50, 0, 250, 300)), (0, 30))
    sheet.paste(sp.convert("RGB"), (1180, 30))
    fgp = Image.open(OUT_STORE / "play-feature-graphic-1024x500.png").resize((512, 250), Image.LANCZOS)
    sheet.paste(fgp, (320, 430))
    wmv = Image.open(OUT_WWW / "dl_watermark.png")
    photo = Image.new("RGB", (330, 220), (92, 108, 124))
    w2 = wmv.resize((140, int(140 * wmv.height / wmv.width)), Image.LANCZOS)
    photo.paste(w2, (330 - 140 - 14, 220 - w2.height - 14), w2)
    sheet.paste(photo, (860, 430))
    save(sheet, OUT_STORE / "preview.png")
    print("Готово.")


if __name__ == "__main__":
    main()
