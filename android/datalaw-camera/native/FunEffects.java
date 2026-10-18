package com.datalaw.camera;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;

import java.util.ArrayList;
import java.util.List;

/**
 * Рендер «смешных» эффектов на Bitmap: искажения лица (mesh-warp), процедурные стикеры, эффекты на весь кадр.
 * Один и тот же код используется для живого превью, записи видео и финального фото.
 */
public class FunEffects {

    public static class Face {
        public float cx, cy, w, h, roll;           // bbox-центр, размеры, наклон (град.)
        public PointF le, re, nose, mouth;         // координаты в кадре (могут быть null)
        PointF lle, lre, lnose, lmouth;            // в локальной системе лица (после поворота)
    }

    public static boolean needsFaces(String id) {
        switch (id) {
            case "bigeyes": case "fat": case "alien": case "nose": case "mouth":
            case "dog": case "clown": case "gent": case "shades": case "crown": case "devil":
                return true;
            default:
                return false;
        }
    }

    private static class Warp {
        final int type; final float cx, cy, r, s;   // type 0: bulge/pinch, 1: swirl
        Warp(int type, float cx, float cy, float r, float s) { this.type = type; this.cx = cx; this.cy = cy; this.r = r; this.s = s; }
    }

    private static final int MX = 36, MY = 48;
    private final float[] verts = new float[(MX + 1) * (MY + 1) * 2];
    private final Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint pBmp = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final Paint pNear = new Paint();
    private final RectF rf = new RectF();
    private final Path path = new Path();
    private Bitmap small;
    private int[] row;
    private int[] thermalLut, postLut;

    /** Рисует результат в dst (того же размера, что src). last — прошлый кадр (для «Призрака»). */
    public void apply(String id, Bitmap src, Bitmap dst, List<Face> faces, Bitmap last) {
        final int w = src.getWidth(), h = src.getHeight();
        Canvas c = new Canvas(dst);
        c.drawBitmap(src, 0, 0, null);
        for (Face f : faces) prep(f);
        List<Warp> ws = new ArrayList<>();

        switch (id) {
            /* ---------- искажения лица ---------- */
            case "bigeyes":
                for (Face f : faces) { ws.add(new Warp(0, f.le.x, f.le.y, .28f * f.w, .95f)); ws.add(new Warp(0, f.re.x, f.re.y, .28f * f.w, .95f)); }
                mesh(c, src, w, h, ws); break;
            case "fat":
                for (Face f : faces) ws.add(new Warp(0, f.cx, f.cy + .05f * f.h, .95f * f.w, .6f));
                mesh(c, src, w, h, ws); break;
            case "alien":
                for (Face f : faces) {
                    ws.add(new Warp(0, f.le.x, f.le.y, .36f * f.w, 1.2f));
                    ws.add(new Warp(0, f.re.x, f.re.y, .36f * f.w, 1.2f));
                    ws.add(new Warp(0, f.cx, f.cy - .4f * f.h, .55f * f.w, .6f));
                    ws.add(new Warp(0, f.cx, f.cy + .42f * f.h, .55f * f.w, -.65f));
                }
                mesh(c, src, w, h, ws); break;
            case "nose":
                for (Face f : faces) ws.add(new Warp(0, f.nose.x, f.nose.y, .32f * f.w, 1.3f));
                mesh(c, src, w, h, ws); break;
            case "mouth":
                for (Face f : faces) ws.add(new Warp(0, f.mouth.x, f.mouth.y, .38f * f.w, 1.1f));
                mesh(c, src, w, h, ws); break;

            /* ---------- стикеры ---------- */
            case "dog": case "clown": case "gent": case "shades": case "crown": case "devil":
                for (Face f : faces) sticker(c, id, f);
                break;

            /* ---------- весь кадр ---------- */
            case "mirror":
                c.save(); c.clipRect(w / 2f, 0, w, h); c.scale(-1, 1, w / 2f, 0); c.drawBitmap(src, 0, 0, null); c.restore();
                break;
            case "kaleido":
                for (int fx = 0; fx < 2; fx++) for (int fy = 0; fy < 2; fy++) {
                    c.save();
                    c.clipRect(fx == 1 ? w / 2f : 0, fy == 1 ? h / 2f : 0, fx == 1 ? w : w / 2f, fy == 1 ? h : h / 2f);
                    c.scale(fx == 1 ? -1 : 1, fy == 1 ? -1 : 1, w / 2f, h / 2f);
                    c.drawBitmap(src, 0, 0, null);
                    c.restore();
                }
                break;
            case "fisheye":
                ws.add(new Warp(0, w / 2f, h / 2f, .75f * Math.max(w, h), 1.15f));
                mesh(c, src, w, h, ws); break;
            case "swirl":
                ws.add(new Warp(1, w / 2f, h / 2f, .65f * Math.min(w, h), 3.2f));
                mesh(c, src, w, h, ws); break;
            case "pixel": {
                int sw = Math.max(8, w / 22), sh = Math.max(8, h / 22);
                if (small == null || small.getWidth() != sw || small.getHeight() != sh)
                    small = Bitmap.createBitmap(sw, sh, Bitmap.Config.ARGB_8888);
                new Canvas(small).drawBitmap(src, new Rect(0, 0, w, h), new Rect(0, 0, sw, sh), pBmp);
                c.drawBitmap(small, new Rect(0, 0, sw, sh), new Rect(0, 0, w, h), pNear);
                break;
            }
            case "thermal":
                remap(dst, true); break;
            case "comic": {
                ColorMatrix cm = new ColorMatrix(); cm.setSaturation(1.7f);
                Paint pp = new Paint(); pp.setColorFilter(new ColorMatrixColorFilter(cm));
                c.drawBitmap(src, 0, 0, pp);
                remap(dst, false); break;
            }
            case "ghost":
                if (last != null && last.getWidth() == w && last.getHeight() == h && last != dst) {
                    c.drawBitmap(last, 0, 0, null);
                    Paint a = new Paint(); a.setAlpha(105);
                    c.drawBitmap(src, 0, 0, a);
                }
                break;
            default: break;
        }
    }

    /* ==================== mesh warp ==================== */

    private void mesh(Canvas c, Bitmap src, int w, int h, List<Warp> ws) {
        if (ws.isEmpty()) return;
        int k = 0;
        for (int j = 0; j <= MY; j++) {
            for (int i = 0; i <= MX; i++) {
                float x = w * (float) i / MX, y = h * (float) j / MY;
                for (Warp s : ws) {
                    float dx = x - s.cx, dy = y - s.cy;
                    float d = (float) Math.sqrt(dx * dx + dy * dy);
                    if (d >= s.r || d == 0) continue;
                    float q = 1 - d / s.r;
                    if (s.type == 0) {
                        float f = 1 + s.s * q * q;
                        x = s.cx + dx * f; y = s.cy + dy * f;
                    } else {
                        float a = s.s * q * q, ca = (float) Math.cos(a), sa = (float) Math.sin(a);
                        x = s.cx + dx * ca - dy * sa; y = s.cy + dx * sa + dy * ca;
                    }
                }
                verts[k++] = x; verts[k++] = y;
            }
        }
        c.drawBitmapMesh(src, MX, MY, verts, 0, null, 0, null);
    }

    /* ==================== пиксельные эффекты ==================== */

    private void remap(Bitmap b, boolean thermal) {
        if (thermalLut == null) buildLuts();
        int w = b.getWidth(), h = b.getHeight();
        if (row == null || row.length < w) row = new int[w];
        for (int y = 0; y < h; y++) {
            b.getPixels(row, 0, w, 0, y, w, 1);
            for (int x = 0; x < w; x++) {
                int px = row[x], r = (px >> 16) & 255, g = (px >> 8) & 255, bl = px & 255;
                if (thermal) row[x] = thermalLut[(r * 77 + g * 150 + bl * 29) >> 8];
                else row[x] = 0xFF000000 | (postLut[r] << 16) | (postLut[g] << 8) | postLut[bl];
            }
            b.setPixels(row, 0, w, 0, y, w, 1);
        }
    }

    private void buildLuts() {
        thermalLut = new int[256]; postLut = new int[256];
        int[][] stops = {{0, 0, 0, 0}, {50, 0, 0, 140}, {100, 150, 0, 170}, {155, 255, 40, 0}, {205, 255, 220, 0}, {255, 255, 255, 255}};
        for (int i = 0; i < 256; i++) {
            int k = 0;
            while (k < stops.length - 2 && i > stops[k + 1][0]) k++;
            float t = (i - stops[k][0]) / (float) (stops[k + 1][0] - stops[k][0]);
            t = Math.max(0f, Math.min(1f, t));
            int r = (int) (stops[k][1] + (stops[k + 1][1] - stops[k][1]) * t);
            int g = (int) (stops[k][2] + (stops[k + 1][2] - stops[k][2]) * t);
            int b = (int) (stops[k][3] + (stops[k + 1][3] - stops[k][3]) * t);
            thermalLut[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
            postLut[i] = Math.min(255, (i * 5 / 256) * 255 / 4);
        }
    }

    /* ==================== геометрия лица ==================== */

    private static void prep(Face f) {
        if (f.le == null) f.le = new PointF(f.cx - .22f * f.w, f.cy - .10f * f.h);
        if (f.re == null) f.re = new PointF(f.cx + .22f * f.w, f.cy - .10f * f.h);
        if (f.nose == null) f.nose = new PointF(f.cx, f.cy + .10f * f.h);
        if (f.mouth == null) f.mouth = new PointF(f.cx, f.cy + .28f * f.h);
        f.lle = loc(f, f.le); f.lre = loc(f, f.re); f.lnose = loc(f, f.nose); f.lmouth = loc(f, f.mouth);
    }

    private static PointF loc(Face f, PointF q) {
        double a = Math.toRadians(f.roll);
        float dx = q.x - f.cx, dy = q.y - f.cy;
        float cs = (float) Math.cos(a), sn = (float) Math.sin(a);
        return new PointF(dx * cs - dy * sn, dx * sn + dy * cs);
    }

    /* ==================== стикеры ==================== */

    private void oval(Canvas c, float cx, float cy, float rx, float ry, int color) {
        p.setStyle(Paint.Style.FILL); p.setColor(color);
        rf.set(cx - rx, cy - ry, cx + rx, cy + ry);
        c.drawOval(rf, p);
    }

    private void sticker(Canvas c, String id, Face f) {
        final float w = f.w, h = f.h;
        c.save();
        c.translate(f.cx, f.cy);
        c.rotate(-f.roll);
        switch (id) {
            case "dog": {
                for (int s = -1; s <= 1; s += 2) {
                    c.save(); c.translate(s * .40f * w, -.46f * h); c.rotate(s * 22);
                    oval(c, 0, 0, .16f * w, .29f * w, 0xFF7B4A24);
                    oval(c, 0, .02f * w, .09f * w, .20f * w, 0xFFE8A0A0);
                    c.restore();
                }
                oval(c, f.lnose.x, f.lnose.y - .01f * w, .10f * w, .07f * w, 0xFF1A1A1A);
                oval(c, f.lnose.x - .03f * w, f.lnose.y - .035f * w, .03f * w, .015f * w, 0xAAFFFFFF);
                p.setStyle(Paint.Style.FILL); p.setColor(0xFFFF6F91);
                rf.set(f.lmouth.x - .07f * w, f.lmouth.y, f.lmouth.x + .07f * w, f.lmouth.y + .19f * w);
                c.drawRoundRect(rf, .06f * w, .06f * w, p);
                p.setStyle(Paint.Style.STROKE); p.setStrokeWidth(.012f * w); p.setColor(0xFFC2185B);
                c.drawLine(f.lmouth.x, f.lmouth.y + .02f * w, f.lmouth.x, f.lmouth.y + .14f * w, p);
                break;
            }
            case "clown": {
                int[] cols = {0xFFFF3D00, 0xFFFFEA00, 0xFF00E676, 0xFF2979FF, 0xFFD500F9, 0xFFFF4081, 0xFFFFA000};
                for (int i = 0; i < cols.length; i++) {
                    double a = Math.toRadians(195 + i * (150.0 / (cols.length - 1)));
                    oval(c, (float) Math.cos(a) * .56f * w, -.12f * h + (float) Math.sin(a) * .66f * h, .17f * w, .17f * w, cols[i]);
                }
                oval(c, f.lnose.x, f.lnose.y, .105f * w, .105f * w, 0xFFE53935);
                oval(c, f.lnose.x - .035f * w, f.lnose.y - .04f * w, .03f * w, .022f * w, 0xBBFFFFFF);
                break;
            }
            case "gent": {
                // цилиндр
                p.setStyle(Paint.Style.FILL); p.setColor(0xFF161616);
                rf.set(-.5f * w, -.56f * h, .5f * w, -.5f * h + .02f * w); c.drawRoundRect(rf, .03f * w, .03f * w, p);
                rf.set(-.28f * w, -.56f * h - .42f * w, .28f * w, -.54f * h); c.drawRoundRect(rf, .04f * w, .04f * w, p);
                p.setColor(0xFFB71C1C);
                rf.set(-.28f * w, -.56f * h - .1f * w, .28f * w, -.54f * h); c.drawRect(rf, p);
                // усы
                for (int s = -1; s <= 1; s += 2) {
                    c.save(); c.translate(f.lnose.x + s * .11f * w, f.lnose.y + .105f * w); c.rotate(-s * 14);
                    oval(c, 0, 0, .14f * w, .05f * w, 0xFF2B1B10);
                    oval(c, s * .13f * w, -.03f * w, .035f * w, .035f * w, 0xFF2B1B10);
                    c.restore();
                }
                // монокль
                PointF m = f.lre.x > f.lle.x ? f.lre : f.lle;
                p.setStyle(Paint.Style.STROKE); p.setStrokeWidth(.02f * w); p.setColor(0xFFFFC107);
                c.drawCircle(m.x, m.y, .11f * w, p);
                c.drawLine(m.x + .08f * w, m.y + .09f * w, m.x + .12f * w, m.y + .45f * w, p);
                break;
            }
            case "shades": {
                float y = (f.lle.y + f.lre.y) / 2f, x0 = Math.min(f.lle.x, f.lre.x) - .18f * w, x1 = Math.max(f.lle.x, f.lre.x) + .18f * w;
                float hh = .19f * w, lw = (x1 - x0) * .45f;
                p.setStyle(Paint.Style.FILL); p.setColor(0xFF0D0D0D);
                rf.set(x0, y - hh * .5f, x0 + lw, y + hh * .6f); c.drawRoundRect(rf, .05f * w, .05f * w, p);
                rf.set(x1 - lw, y - hh * .5f, x1, y + hh * .6f); c.drawRoundRect(rf, .05f * w, .05f * w, p);
                rf.set(x0 + lw - .01f * w, y - hh * .4f, x1 - lw + .01f * w, y - hh * .2f); c.drawRect(rf, p);
                p.setStyle(Paint.Style.STROKE); p.setStrokeWidth(.03f * w);
                c.drawLine(x0, y - hh * .35f, x0 - .12f * w, y - hh * .25f, p);
                c.drawLine(x1, y - hh * .35f, x1 + .12f * w, y - hh * .25f, p);
                p.setColor(0x66FFFFFF); p.setStrokeWidth(.018f * w);
                c.drawLine(x0 + .06f * w, y - hh * .3f, x0 + .15f * w, y + hh * .4f, p);
                c.drawLine(x1 - lw + .06f * w, y - hh * .3f, x1 - lw + .15f * w, y + hh * .4f, p);
                break;
            }
            case "crown": {
                path.reset();
                path.moveTo(-.30f * w, -.50f * h); path.lineTo(-.34f * w, -.80f * h); path.lineTo(-.16f * w, -.63f * h);
                path.lineTo(0, -.88f * h); path.lineTo(.16f * w, -.63f * h); path.lineTo(.34f * w, -.80f * h);
                path.lineTo(.30f * w, -.50f * h); path.close();
                p.setStyle(Paint.Style.FILL); p.setColor(0xFFFFC107); c.drawPath(path, p);
                p.setStyle(Paint.Style.STROKE); p.setStrokeWidth(.018f * w); p.setColor(0xFFB8860B); c.drawPath(path, p);
                oval(c, 0, -.57f * h, .035f * w, .035f * w, 0xFFE53935);
                oval(c, -.17f * w, -.55f * h, .03f * w, .03f * w, 0xFF2979FF);
                oval(c, .17f * w, -.55f * h, .03f * w, .03f * w, 0xFF00C853);
                break;
            }
            case "devil": {
                for (int s = -1; s <= 1; s += 2) {
                    path.reset();
                    path.moveTo(s * .40f * w, -.40f * h);
                    path.quadTo(s * .55f * w, -.68f * h, s * .30f * w, -.84f * h);
                    path.quadTo(s * .27f * w, -.62f * h, s * .16f * w, -.48f * h);
                    path.close();
                    p.setStyle(Paint.Style.FILL); p.setColor(0xFFD50000); c.drawPath(path, p);
                }
                path.reset();
                path.moveTo(-.07f * w, .44f * h); path.lineTo(.07f * w, .44f * h); path.lineTo(0, .64f * h); path.close();
                p.setStyle(Paint.Style.FILL); p.setColor(0xFF3E0A0A); c.drawPath(path, p);
                break;
            }
            default: break;
        }
        c.restore();
    }
}
