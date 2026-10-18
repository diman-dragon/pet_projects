# Generates launcher icons, splash, Play 512 icon and feature graphic from one shared drawing. Run: python3 resources/make-icons.py
import os
from PIL import Image, ImageDraw, ImageFont
BG, PAGE, FOLD, INK = (59, 39, 22), (241, 228, 200), (201, 178, 135), (59, 39, 22)
RES = 'android/app/src/main/res'

def draw(size, shape, pad=1.0, ss=4):
    """shape: 'square' (store) | 'rounded' (legacy) | 'round'. Page geometry mirrors drawable/ic_launcher_foreground.xml (108 grid)."""
    S = size * ss
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if shape == 'round': d.ellipse([0, 0, S - 1, S - 1], fill=BG)
    elif shape == 'rounded': d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * .18), fill=BG)
    else: d.rectangle([0, 0, S, S], fill=BG)
    k = S / 108 * pad
    o = S / 2 - 54 * k
    P = lambda x, y: (o + x * k, o + y * k)
    d.polygon([P(33, 26), P(63, 26), P(75, 38), P(75, 82), P(33, 82)], fill=PAGE)
    d.polygon([P(63, 26), P(63, 38), P(75, 38)], fill=FOLD)
    for y, x2 in ((48, 68), (56, 68), (64, 58)):
        d.rectangle([*P(40, y), *P(x2, y + 3)], fill=INK)
    return img.resize((size, size), Image.LANCZOS)

dens = {'mdpi': 48, 'hdpi': 72, 'xhdpi': 96, 'xxhdpi': 144, 'xxxhdpi': 192}
for name, px in dens.items():
    d = f'{RES}/mipmap-{name}'
    draw(px, 'rounded', pad=1.25).save(f'{d}/ic_launcher.png')
    draw(px, 'round', pad=1.25).save(f'{d}/ic_launcher_round.png')
    fg = f'{d}/ic_launcher_foreground.png'
    if os.path.exists(fg): os.remove(fg)
for root, _, fs in os.walk(RES):
    if 'splash.png' in fs:
        p = f'{root}/splash.png'
        w, h = Image.open(p).size
        Image.new('RGB', (w, h), BG).save(p)
draw(512, 'square', pad=1.25).convert('RGB').save('resources/play_store_icon_512.png')
fg = Image.new('RGB', (1024, 500), BG)
ic = draw(260, 'rounded', pad=1.25)
fg.paste(ic, (120, 120), ic)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 110)
except OSError: font = ImageFont.load_default()
ImageDraw.Draw(fg).text((440, 190), 'Folio', font=font, fill=PAGE)
fg.save('resources/feature_graphic_1024x500.png')
print('icons ok')
