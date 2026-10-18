package com.datalaw.camera;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.util.Size;
import android.view.Surface;
import android.view.View;

import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;

import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.IntConsumer;

/**
 * Конвейер «весёлых» эффектов: ImageAnalysis -> ML Kit Face Detection -> FunEffects -> FunView
 * (+ запись обработанных кадров в MediaRecorder через Surface).
 */
public class FunPipeline {

    /** Вид, который рисует обработанные кадры поверх (вместо) PreviewView. */
    public static class FunView extends View {
        private Bitmap bmp;
        private final Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        private final Rect src = new Rect();
        private final RectF dst = new RectF();

        public FunView(Context c) { super(c); }

        public void show(Bitmap b) { bmp = b; postInvalidateOnAnimation(); }

        @Override
        protected void onDraw(Canvas canvas) {
            Bitmap b = bmp;
            if (b == null || getWidth() == 0 || getHeight() == 0) return;
            float vr = (float) getWidth() / getHeight(), br = (float) b.getWidth() / b.getHeight();
            int sw = b.getWidth(), sh = b.getHeight();
            if (br > vr) sw = Math.round(sh * vr); else sh = Math.round(sw / vr);   // FILL_CENTER
            int sx = (b.getWidth() - sw) / 2, sy = (b.getHeight() - sh) / 2;
            src.set(sx, sy, sx + sw, sy + sh);
            dst.set(0, 0, getWidth(), getHeight());
            canvas.drawBitmap(b, src, dst, paint);
        }
    }

    private final Context ctx;
    public final FunView view;
    public volatile String effect = "none";
    public volatile boolean front = false;
    public volatile IntConsumer facesCb;

    private final ExecutorService exec = Executors.newSingleThreadExecutor();
    private final FunEffects fx = new FunEffects();
    private final FaceDetector fast;
    private FaceDetector accurate;
    private final Paint pBmp = new Paint(Paint.FILTER_BITMAP_FLAG);

    private Bitmap raw, up, outA, outB;
    private int flip = 0;
    private long lastNotify = 0;
    private int lastFaces = -1;

    // запись
    private volatile MediaRecorder mr;
    private volatile Surface recSurface;
    private ParcelFileDescriptor pfd;
    private Uri recUri;
    private int recW, recH;

    public FunPipeline(Context ctx) {
        this.ctx = ctx;
        this.view = new FunView(ctx);
        this.fast = FaceDetection.getClient(new FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setMinFaceSize(0.12f)
            .build());
    }

    public ImageAnalysis build() {
        ImageAnalysis ia = new ImageAnalysis.Builder()
            .setResolutionSelector(new ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(new ResolutionStrategy(new Size(960, 720),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                .build())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build();
        ia.setAnalyzer(exec, this::analyze);
        return ia;
    }

    /* ==================== кадры ==================== */

    private void analyze(ImageProxy image) {
        try {
            ImageProxy.PlaneProxy pl = image.getPlanes()[0];
            final int w = image.getWidth(), h = image.getHeight();
            final int stride = pl.getRowStride() / pl.getPixelStride();
            if (raw == null || raw.getWidth() != stride || raw.getHeight() != h) {
                raw = Bitmap.createBitmap(stride, h, Bitmap.Config.ARGB_8888);
            }
            ByteBuffer buf = pl.getBuffer();
            buf.rewind();
            raw.copyPixelsFromBuffer(buf);

            // поворот в «вертикальный» вид + зеркало для фронтальной камеры
            Matrix m = new Matrix();
            m.postRotate(image.getImageInfo().getRotationDegrees());
            RectF rr = new RectF(0, 0, w, h);
            m.mapRect(rr);
            m.postTranslate(-rr.left, -rr.top);
            final int ow = Math.round(rr.width()), oh = Math.round(rr.height());
            if (front) m.postScale(-1, 1, ow / 2f, 0);
            if (up == null || up.getWidth() != ow || up.getHeight() != oh) {
                up = Bitmap.createBitmap(ow, oh, Bitmap.Config.ARGB_8888);
                outA = Bitmap.createBitmap(ow, oh, Bitmap.Config.ARGB_8888);
                outB = Bitmap.createBitmap(ow, oh, Bitmap.Config.ARGB_8888);
            }
            Canvas cu = new Canvas(up);
            cu.save();
            cu.concat(m);
            cu.drawBitmap(raw, new Rect(0, 0, w, h), new RectF(0, 0, w, h), pBmp);
            cu.restore();

            final String id = effect;
            List<FunEffects.Face> faces = FunEffects.needsFaces(id) ? detect(up, fast, 1f) : new ArrayList<>();
            notifyFaces(FunEffects.needsFaces(id) ? faces.size() : -1);

            Bitmap dst = (flip++ % 2 == 0) ? outA : outB;
            Bitmap last = (dst == outA) ? outB : outA;
            fx.apply(id, up, dst, faces, last);
            view.show(dst);

            Surface rs = recSurface;
            if (rs != null) {
                try {
                    Canvas c = rs.lockHardwareCanvas();
                    c.drawBitmap(dst, new Rect(0, 0, dst.getWidth(), dst.getHeight()), new Rect(0, 0, recW, recH), pBmp);
                    rs.unlockCanvasAndPost(c);
                } catch (Exception ignored) { }
            }
        } catch (Throwable t) {
            // пропускаем кадр
        } finally {
            image.close();
        }
    }

    private void notifyFaces(int n) {
        long now = System.currentTimeMillis();
        if (n != lastFaces || now - lastNotify > 1500) {
            if (n != lastFaces || now - lastNotify > 400) {
                lastFaces = n; lastNotify = now;
                IntConsumer cb = facesCb;
                if (cb != null) cb.accept(n);
            }
        }
    }

    private List<FunEffects.Face> detect(Bitmap bmp, FaceDetector det, float k) throws Exception {
        List<Face> found = Tasks.await(det.process(InputImage.fromBitmap(bmp, 0)));
        List<FunEffects.Face> out = new ArrayList<>();
        for (Face f : found) {
            FunEffects.Face o = new FunEffects.Face();
            Rect b = f.getBoundingBox();
            o.cx = b.centerX() * k; o.cy = b.centerY() * k; o.w = b.width() * k; o.h = b.height() * k;
            o.roll = f.getHeadEulerAngleZ();
            o.le = lm(f, FaceLandmark.LEFT_EYE, k);
            o.re = lm(f, FaceLandmark.RIGHT_EYE, k);
            o.nose = lm(f, FaceLandmark.NOSE_BASE, k);
            PointF mb = lm(f, FaceLandmark.MOUTH_BOTTOM, k);
            PointF ml = lm(f, FaceLandmark.MOUTH_LEFT, k), mrr = lm(f, FaceLandmark.MOUTH_RIGHT, k);
            if (ml != null && mrr != null && mb != null) o.mouth = new PointF((ml.x + mrr.x) / 2f, (ml.y + mrr.y + mb.y * 2f) / 4f);
            else o.mouth = mb;
            out.add(o);
        }
        return out;
    }

    private static PointF lm(Face f, int type, float k) {
        FaceLandmark l = f.getLandmark(type);
        if (l == null) return null;
        PointF p = l.getPosition();
        return new PointF(p.x * k, p.y * k);
    }

    /* ==================== финальное фото ==================== */

    /** Зеркалит (для фронталки) и применяет эффект к полноразмерному кадру. Исходный bmp освобождается. */
    public Bitmap process(Bitmap bmp, String id, boolean mirror) throws Exception {
        if (mirror) {
            Matrix m = new Matrix();
            m.postScale(-1, 1, bmp.getWidth() / 2f, 0);
            Bitmap r = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), m, true);
            if (r != bmp) bmp.recycle();
            bmp = r;
        }
        if (id == null || "none".equals(id)) return bmp;
        List<FunEffects.Face> faces = new ArrayList<>();
        if (FunEffects.needsFaces(id)) {
            if (accurate == null) {
                accurate = FaceDetection.getClient(new FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                    .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                    .build());
            }
            float s = Math.min(1f, 960f / Math.max(bmp.getWidth(), bmp.getHeight()));
            Bitmap small = Bitmap.createScaledBitmap(bmp, Math.max(1, Math.round(bmp.getWidth() * s)),
                Math.max(1, Math.round(bmp.getHeight() * s)), true);
            faces = detect(small, accurate, 1f / s);
            small.recycle();
        }
        Bitmap out = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
        new FunEffects().apply(id, bmp, out, faces, null);
        bmp.recycle();
        return out;
    }

    /* ==================== запись видео с эффектами ==================== */

    public void startRecording(String name, boolean audio) throws Exception {
        if (outA == null) throw new Exception("нет кадров с камеры");
        if (mr != null) throw new Exception("уже идёт запись");
        recW = outA.getWidth() & ~1;
        recH = outA.getHeight() & ~1;

        ContentResolver cr = ctx.getContentResolver();
        ContentValues v = new ContentValues();
        v.put(MediaStore.MediaColumns.DISPLAY_NAME, name + ".mp4");
        v.put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4");
        v.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DCIM + "/DataLaw");
        v.put(MediaStore.MediaColumns.IS_PENDING, 1);
        recUri = cr.insert(MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), v);
        if (recUri == null) throw new Exception("MediaStore insert failed");
        pfd = cr.openFileDescriptor(recUri, "w");

        @SuppressWarnings("deprecation")
        MediaRecorder m = new MediaRecorder();
        try {
            if (audio) m.setAudioSource(MediaRecorder.AudioSource.MIC);
            m.setVideoSource(MediaRecorder.VideoSource.SURFACE);
            m.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            m.setOutputFile(pfd.getFileDescriptor());
            m.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
            if (audio) {
                m.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
                m.setAudioEncodingBitRate(128_000);
                m.setAudioSamplingRate(44_100);
            }
            m.setVideoSize(recW, recH);
            m.setVideoFrameRate(30);
            m.setVideoEncodingBitRate(8_000_000);
            m.prepare();
            recSurface = m.getSurface();
            m.start();
            mr = m;
        } catch (Exception e) {
            try { m.release(); } catch (Exception ignored) { }
            cleanupFailed();
            throw e;
        }
    }

    public Uri stopRecording() throws Exception {
        MediaRecorder m = mr;
        if (m == null) throw new Exception("не идёт запись");
        mr = null; recSurface = null;
        try {
            m.stop();
        } catch (RuntimeException e) {
            try { m.release(); } catch (Exception ignored) { }
            cleanupFailed();
            throw new Exception("запись слишком короткая");
        }
        m.release();
        try { if (pfd != null) pfd.close(); } catch (Exception ignored) { }
        ContentValues u = new ContentValues();
        u.put(MediaStore.MediaColumns.IS_PENDING, 0);
        ctx.getContentResolver().update(recUri, u, null, null);
        Uri r = recUri; recUri = null; pfd = null;
        return r;
    }

    private void cleanupFailed() {
        try { if (pfd != null) pfd.close(); } catch (Exception ignored) { }
        if (recUri != null) {
            try { ctx.getContentResolver().delete(recUri, null, null); } catch (Exception ignored) { }
        }
        recUri = null; pfd = null; mr = null; recSurface = null;
    }

    public boolean isRecording() { return mr != null; }

    public void release() {
        try { if (mr != null) stopRecording(); } catch (Exception ignored) { }
        exec.shutdown();
        fast.close();
        if (accurate != null) accurate.close();
    }
}
