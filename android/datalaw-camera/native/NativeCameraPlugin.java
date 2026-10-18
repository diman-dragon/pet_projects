package com.datalaw.camera;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CaptureRequest;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.os.Handler;
import android.os.Looper;
import android.util.Base64;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.OptIn;
import androidx.camera.camera2.interop.Camera2CameraControl;
import androidx.camera.camera2.interop.Camera2CameraInfo;
import androidx.camera.camera2.interop.CaptureRequestOptions;
import androidx.camera.camera2.interop.ExperimentalCamera2Interop;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraInfo;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.CameraState;
import androidx.camera.core.ExposureState;
import androidx.camera.core.FocusMeteringAction;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.MeteringPoint;
import androidx.camera.core.Preview;
import androidx.camera.core.UseCase;
import androidx.camera.core.ZoomState;
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.video.FallbackStrategy;
import androidx.camera.video.MediaStoreOutputOptions;
import androidx.camera.video.PendingRecording;
import androidx.camera.video.Quality;
import androidx.camera.video.QualitySelector;
import androidx.camera.video.Recorder;
import androidx.camera.video.Recording;
import androidx.camera.video.VideoCapture;
import androidx.camera.video.VideoRecordEvent;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.Observer;

import com.getcapacitor.JSObject;
import com.getcapacitor.PermissionState;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;
import com.getcapacitor.annotation.PermissionCallback;
import com.google.android.gms.tasks.Tasks;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.segmentation.Segmentation;
import com.google.mlkit.vision.segmentation.SegmentationMask;
import com.google.mlkit.vision.segmentation.Segmenter;
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Нативная камера на CameraX: превью лежит ПОД прозрачным WebView.
 * Фото/видео сохраняются сразу в DCIM/DataLaw. Ручные ISO/выдержка — через Camera2 interop.
 * Портрет — ML Kit Selfie Segmentation + размытие фона.
 */
@CapacitorPlugin(
    name = "NativeCamera",
    permissions = {
        @Permission(strings = {Manifest.permission.CAMERA}, alias = "camera"),
        @Permission(strings = {Manifest.permission.RECORD_AUDIO}, alias = "microphone")
    }
)
public class NativeCameraPlugin extends Plugin {

    private ProcessCameraProvider provider;
    private PreviewView previewView;
    private Camera camera;
    private Preview preview;
    private ImageCapture imageCapture;
    private VideoCapture<Recorder> videoCapture;
    private Recording recording;
    private PluginCall stopCall;
    private boolean front = false;
    private Segmenter segmenter;
    private FunPipeline fun;
    private boolean funMode = false;
    private String funId = "none";
    private static final String TAG = "DataLawCam";
    private final ExecutorService io = Executors.newSingleThreadExecutor();

    private void ui(Runnable r) { getActivity().runOnUiThread(r); }

    /* ====================== start / stop ====================== */

    /** Быстрая проверка, что плагин зарегистрирован и отвечает. */
    @PluginMethod
    public void ping(PluginCall call) {
        JSObject o = new JSObject();
        o.put("ok", true);
        o.put("sdk", android.os.Build.VERSION.SDK_INT);
        o.put("device", android.os.Build.MANUFACTURER + " " + android.os.Build.MODEL);
        call.resolve(o);
    }

    /**
     * Запуск камеры. Разрешения запрашиваются отдельно (JS вызывает requestPermissions()),
     * поэтому здесь нет сохранённых вызовов и колбэков — start() всегда отвечает.
     */
    @PluginMethod
    public void start(PluginCall call) {
        if (getPermissionState("camera") != PermissionState.GRANTED) {
            call.reject("PERMISSION_DENIED");
            return;
        }
        doStart(call);
    }

    private void doStart(PluginCall call) {
        final String mode = call.getString("mode", "photo");
        final boolean wantFront = "user".equals(call.getString("facing", "environment"));
        final int res = call.getInt("res", 1080);
        final String css = call.getString("css", "");
        final double x = call.getDouble("x", 0.0), y = call.getDouble("y", 0.0);
        final double w = call.getDouble("w", 360.0), h = call.getDouble("h", 480.0);
        ui(() -> {
            try {
                ensurePreview();
                applyRect(x, y, w, h);
                applyFilter(css);
                final ListenableFuture<ProcessCameraProvider> f = ProcessCameraProvider.getInstance(getContext());
                f.addListener(() -> {
                    try {
                        provider = f.get();
                        startBound(call, mode, wantFront, res, false);
                    } catch (Throwable t) {
                        Log.e(TAG, "bind failed", t);
                        call.reject("CameraX bind: " + t);
                    }
                }, ContextCompat.getMainExecutor(getContext()));
            } catch (Throwable t) {
                Log.e(TAG, "start failed", t);
                call.reject("CameraX start: " + t);
            }
        });
    }

    private void ensurePreview() {
        if (previewView != null) return;
        ViewGroup parent = (ViewGroup) getBridge().getWebView().getParent();
        previewView = new PreviewView(getContext());
        previewView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);
        parent.setBackgroundColor(Color.BLACK);
        parent.addView(previewView, 0, new ViewGroup.LayoutParams(1, 1));
        getBridge().getWebView().setBackgroundColor(Color.TRANSPARENT);
    }

    /** Конвейер эффектов создаётся только когда пользователь открыл режим «Эффекты». */
    private void ensureFun() {
        if (fun != null) return;
        ViewGroup parent = (ViewGroup) getBridge().getWebView().getParent();
        fun = new FunPipeline(getContext());
        fun.view.setVisibility(View.GONE);
        fun.facesCb = n -> notifyListeners("faces", new JSObject().put("n", n));
        parent.addView(fun.view, 1, new ViewGroup.LayoutParams(1, 1));
        applyRect(lastX, lastY, lastW, lastH);
    }

    private double lastX = 0, lastY = 0, lastW = 360, lastH = 480;

    private void applyRect(double x, double y, double w, double h) {
        lastX = x; lastY = y; lastW = w; lastH = h;
        if (previewView == null) return;
        float d = getContext().getResources().getDisplayMetrics().density;
        ViewGroup.LayoutParams lp = previewView.getLayoutParams();
        lp.width = Math.max(1, Math.round((float) w * d));
        lp.height = Math.max(1, Math.round((float) h * d));
        previewView.setLayoutParams(lp);
        // CSS-координаты считаются от левого-верхнего угла WebView (на случай, если он смещён родителем)
        View wv = getBridge().getWebView();
        float ox = wv.getLeft() + wv.getTranslationX(), oy = wv.getTop() + wv.getTranslationY();
        previewView.setX(ox + (float) x * d);
        previewView.setY(oy + (float) y * d);
        if (fun != null) {
            ViewGroup.LayoutParams l2 = fun.view.getLayoutParams();
            l2.width = lp.width; l2.height = lp.height;
            fun.view.setLayoutParams(l2);
            fun.view.setX(ox + (float) x * d);
            fun.view.setY(oy + (float) y * d);
        }
    }

    private void bind(String mode, boolean wantFront, int res, boolean safe) {
        provider.unbindAll();
        recording = null;
        front = wantFront;
        CameraSelector sel = new CameraSelector.Builder()
            .requireLensFacing(front ? CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK)
            .build();
        boolean video = "video".equals(mode);
        funMode = "fun".equals(mode);

        AspectRatioStrategy ars = video
            ? AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY
            : AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY;
        preview = new Preview.Builder()
            .setResolutionSelector(new ResolutionSelector.Builder().setAspectRatioStrategy(ars).build())
            .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        UseCase second;
        if (video) {
            Quality q = res >= 2160 ? Quality.UHD : (res >= 1080 ? Quality.FHD : Quality.HD);
            Recorder recorder = new Recorder.Builder()
                .setQualitySelector(QualitySelector.from(q, FallbackStrategy.lowerQualityOrHigherThan(q)))
                .build();
            videoCapture = VideoCapture.withOutput(recorder);
            imageCapture = null;
            second = videoCapture;
        } else {
            ResolutionSelector.Builder rsb = new ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY);
            if (!safe) {
                rsb.setResolutionStrategy(new ResolutionStrategy(new Size(4000, 3000),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER));
            }
            imageCapture = new ImageCapture.Builder()
                .setCaptureMode(!safe && !"burst".equals(mode)
                    ? ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY
                    : ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setResolutionSelector(rsb.build())
                .build();
            videoCapture = null;
            second = imageCapture;
        }
        if (funMode) {
            ensureFun();
            previewView.setVisibility(View.INVISIBLE);
            fun.view.setVisibility(View.VISIBLE);
            fun.front = front;
            fun.effect = funId;
            camera = provider.bindToLifecycle((LifecycleOwner) getActivity(), sel, fun.build(), second);
        } else {
            previewView.setVisibility(View.VISIBLE);
            if (fun != null) fun.view.setVisibility(View.GONE);
            camera = provider.bindToLifecycle((LifecycleOwner) getActivity(), sel, preview, second);
        }
    }

    /**
     * Привязка + ожидание РЕАЛЬНОГО открытия камеры. Если камера не открылась (ошибка конфигурации потоков,
     * «камера занята», тайм-аут 6 с) — одна повторная попытка с безопасными настройками, затем reject,
     * после чего JS переходит на режим совместимости. Никакого вечного ожидания.
     */
    private void startBound(final PluginCall call, final String mode, final boolean wantFront, final int res, final boolean safe) {
        try {
            bind(mode, wantFront, res, safe);
        } catch (Throwable t) {
            Log.e(TAG, "bind failed (safe=" + safe + ")", t);
            if (!safe) { startBound(call, mode, wantFront, res, true); return; }
            call.reject("CameraX bind: " + t);
            return;
        }
        new OpenWaiter(call, camera.getCameraInfo().getCameraState(), () -> {
            if (!safe) startBound(call, mode, wantFront, res, true);
            else call.reject("CameraX: камера не открылась");
        }).begin();
    }

    private class OpenWaiter implements Observer<CameraState>, Runnable {
        private final PluginCall call;
        private final LiveData<CameraState> live;
        private final Runnable onFail;
        private final Handler h = new Handler(Looper.getMainLooper());
        private boolean done = false;

        OpenWaiter(PluginCall call, LiveData<CameraState> live, Runnable onFail) {
            this.call = call; this.live = live; this.onFail = onFail;
        }

        void begin() {
            live.observe((LifecycleOwner) getActivity(), this);
            h.postDelayed(this, 6000);
        }

        @Override
        public void onChanged(CameraState st) {
            if (done || st == null) return;
            if (st.getType() == CameraState.Type.OPEN) { finish(true); return; }
            CameraState.StateError err = st.getError();
            if (err != null) {
                int c = err.getCode();
                Log.w(TAG, "camera state error " + c);
                // 4 STREAM_CONFIG, 5 CAMERA_DISABLED, 6 FATAL, 7 DO_NOT_DISTURB — ждать бессмысленно
                if (c >= 4 && c <= 7) finish(false);
            }
        }

        @Override
        public void run() {
            if (!done) { Log.w(TAG, "camera open timeout"); finish(false); }
        }

        private void finish(boolean ok) {
            done = true;
            h.removeCallbacks(this);
            live.removeObserver(this);
            if (ok) {
                try { call.resolve(caps()); } catch (Throwable t) { call.reject("CameraX caps: " + t); }
            } else {
                onFail.run();
            }
        }
    }

    @OptIn(markerClass = ExperimentalCamera2Interop.class)
    private JSObject caps() {
        JSObject o = new JSObject();
        CameraInfo info = camera.getCameraInfo();
        o.put("zoomMin", 1.0); o.put("zoomMax", 1.0); o.put("hasEv", false); o.put("torch", false); o.put("front", front);
        try {
        ZoomState zs = info.getZoomState().getValue();
        o.put("zoomMin", zs != null ? (double) zs.getMinZoomRatio() : 1.0);
        o.put("zoomMax", zs != null ? (double) Math.min(zs.getMaxZoomRatio(), 20f) : 1.0);
        o.put("torch", info.hasFlashUnit());
        o.put("front", front);
        ExposureState es = info.getExposureState();
        o.put("hasEv", es.isExposureCompensationSupported());
        Range<Integer> r = es.getExposureCompensationRange();
        double step = es.getExposureCompensationStep().doubleValue();
        o.put("evMin", r.getLower() * step);
        o.put("evMax", r.getUpper() * step);
        o.put("evStep", step);
        try {
            Camera2CameraInfo c2 = Camera2CameraInfo.from(info);
            Range<Integer> iso = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE);
            Range<Long> exp = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE);
            int[] capab = c2.getCameraCharacteristic(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
            boolean manual = false;
            if (capab != null) {
                for (int c : capab) {
                    if (c == CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) manual = true;
                }
            }
            if (manual && iso != null && exp != null) {
                o.put("pro", true);
                o.put("isoMin", (double) iso.getLower());
                o.put("isoMax", (double) iso.getUpper());
                o.put("expMin", exp.getLower() / 1e9);
                o.put("expMax", exp.getUpper() / 1e9);
            }
        } catch (Exception ignored) { }
        } catch (Throwable t) {
            Log.w(TAG, "caps failed", t);
        }
        return o;
    }

    @PluginMethod
    public void stop(PluginCall call) {
        ui(() -> {
            try {
                if (recording != null) { recording.stop(); recording = null; }
                if (provider != null) provider.unbindAll();
                if (fun != null) {
                    ((ViewGroup) fun.view.getParent()).removeView(fun.view);
                    fun.release();
                    fun = null;
                }
                funMode = false;
                if (previewView != null) {
                    ((ViewGroup) previewView.getParent()).removeView(previewView);
                    previewView = null;
                }
            } catch (Exception ignored) { }
            call.resolve();
        });
    }

    @PluginMethod
    public void setRect(PluginCall call) {
        final double x = call.getDouble("x", 0.0), y = call.getDouble("y", 0.0);
        final double w = call.getDouble("w", 360.0), h = call.getDouble("h", 480.0);
        ui(() -> { applyRect(x, y, w, h); call.resolve(); });
    }

    /* ====================== управление ====================== */

    @PluginMethod
    public void setZoom(PluginCall call) {
        final float v = call.getFloat("value", 1f);
        ui(() -> { if (camera != null) camera.getCameraControl().setZoomRatio(v); call.resolve(); });
    }

    @PluginMethod
    public void focus(PluginCall call) {
        final float x = call.getFloat("x", 0.5f), y = call.getFloat("y", 0.5f);
        ui(() -> {
            if (camera != null && previewView != null && !funMode) {
                MeteringPoint p = previewView.getMeteringPointFactory()
                    .createPoint(x * previewView.getWidth(), y * previewView.getHeight());
                camera.getCameraControl().startFocusAndMetering(
                    new FocusMeteringAction.Builder(p).setAutoCancelDuration(3, TimeUnit.SECONDS).build());
            }
            call.resolve();
        });
    }

    @PluginMethod
    public void torch(PluginCall call) {
        final boolean on = Boolean.TRUE.equals(call.getBoolean("on", false));
        ui(() -> { if (camera != null) camera.getCameraControl().enableTorch(on); call.resolve(); });
    }

    @PluginMethod
    public void ev(PluginCall call) {
        final double v = call.getDouble("value", 0.0);
        ui(() -> {
            if (camera != null) {
                double step = camera.getCameraInfo().getExposureState().getExposureCompensationStep().doubleValue();
                if (step > 0) camera.getCameraControl().setExposureCompensationIndex((int) Math.round(v / step));
            }
            call.resolve();
        });
    }

    /** Ручной режим: ISO + выдержка. on=false возвращает автоэкспозицию. */
    @PluginMethod
    public void manual(PluginCall call) {
        final boolean on = Boolean.TRUE.equals(call.getBoolean("on", false));
        final int iso = call.getInt("iso", 200);
        final long expNs = (long) (double) call.getDouble("expNs", 16666666.0);
        ui(() -> {
            try { applyManual(on, iso, expNs); call.resolve(); }
            catch (Exception e) { call.reject(e.getMessage()); }
        });
    }

    @OptIn(markerClass = ExperimentalCamera2Interop.class)
    private void applyManual(boolean on, int iso, long expNs) {
        if (camera == null) return;
        Camera2CameraControl c2 = Camera2CameraControl.from(camera.getCameraControl());
        CaptureRequestOptions.Builder b = new CaptureRequestOptions.Builder();
        if (on) {
            b.setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF);
            b.setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, iso);
            b.setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, expNs);
            b.setCaptureRequestOption(CaptureRequest.SENSOR_FRAME_DURATION, Math.max(expNs, 33_333_333L));
        }
        c2.setCaptureRequestOptions(b.build()); // пустой набор = обратно в авто
    }

    @PluginMethod
    public void funEffect(PluginCall call) {
        funId = call.getString("id", "none");
        if (fun != null) fun.effect = funId;
        call.resolve();
    }

    @PluginMethod
    public void filter(PluginCall call) {
        final String css = call.getString("css", "");
        ui(() -> { applyFilter(css); call.resolve(); });
    }

    private void applyFilter(String css) {
        if (previewView == null) return;
        ColorMatrix m = cssToMatrix(css);
        if (m == null) {
            previewView.setLayerType(View.LAYER_TYPE_NONE, null);
        } else {
            Paint p = new Paint();
            p.setColorFilter(new ColorMatrixColorFilter(m));
            previewView.setLayerType(View.LAYER_TYPE_HARDWARE, p);
        }
    }

    /** Те же CSS-фильтры, что и в вебе, переведённые в ColorMatrix. */
    private static ColorMatrix cssToMatrix(String css) {
        if (css == null || css.trim().isEmpty()) return null;
        Matcher m = Pattern.compile("([a-z-]+)\\(([^)]*)\\)").matcher(css);
        ColorMatrix acc = new ColorMatrix();
        boolean any = false;
        while (m.find()) {
            String fn = m.group(1);
            float v;
            try { v = Float.parseFloat(m.group(2).trim().replace("deg", "")); }
            catch (Exception e) { continue; }
            ColorMatrix c = new ColorMatrix();
            switch (fn) {
                case "saturate": c.setSaturation(v); break;
                case "grayscale": c.setSaturation(1 - v); break;
                case "brightness": c.setScale(v, v, v, 1); break;
                case "contrast": {
                    float t = 127.5f * (1 - v);
                    c.set(new float[]{v, 0, 0, 0, t, 0, v, 0, 0, t, 0, 0, v, 0, t, 0, 0, 0, 1, 0});
                    break;
                }
                case "sepia": {
                    float k = 1 - v;
                    c.set(new float[]{
                        0.393f + 0.607f * k, 0.769f - 0.769f * k, 0.189f - 0.189f * k, 0, 0,
                        0.349f - 0.349f * k, 0.686f + 0.314f * k, 0.168f - 0.168f * k, 0, 0,
                        0.272f - 0.272f * k, 0.534f - 0.534f * k, 0.131f + 0.869f * k, 0, 0,
                        0, 0, 0, 1, 0});
                    break;
                }
                case "hue-rotate": {
                    double a = Math.toRadians(v);
                    float cs = (float) Math.cos(a), sn = (float) Math.sin(a);
                    c.set(new float[]{
                        0.213f + cs * 0.787f - sn * 0.213f, 0.715f - cs * 0.715f - sn * 0.715f, 0.072f - cs * 0.072f + sn * 0.928f, 0, 0,
                        0.213f - cs * 0.213f + sn * 0.143f, 0.715f + cs * 0.285f + sn * 0.140f, 0.072f - cs * 0.072f - sn * 0.283f, 0, 0,
                        0.213f - cs * 0.213f - sn * 0.787f, 0.715f - cs * 0.715f + sn * 0.715f, 0.072f + cs * 0.928f + sn * 0.072f, 0, 0,
                        0, 0, 0, 1, 0});
                    break;
                }
                default: continue;
            }
            acc.postConcat(c);
            any = true;
        }
        return any ? acc : null;
    }

    /* ====================== фото ====================== */

    @PluginMethod
    public void photo(PluginCall call) {
        if (imageCapture == null) { call.reject("not in photo mode"); return; }
        final double aspect = call.getDouble("aspect", 0.75);
        final String css = call.getString("css", "");
        final boolean flash = Boolean.TRUE.equals(call.getBoolean("flash", false));
        final boolean portrait = Boolean.TRUE.equals(call.getBoolean("portrait", false));
        final int blur = call.getInt("blur", 2);
        final String name = call.getString("name", "DataLaw_" + System.currentTimeMillis() + ".jpg");
        final String effect = call.getString("effect", funId);
        final boolean watermark = Boolean.TRUE.equals(call.getBoolean("watermark", false));
        ui(() -> {
            imageCapture.setFlashMode(flash ? ImageCapture.FLASH_MODE_ON : ImageCapture.FLASH_MODE_OFF);
            imageCapture.takePicture(io, new ImageCapture.OnImageCapturedCallback() {
                @Override
                public void onCaptureSuccess(ImageProxy image) {
                    try {
                        call.resolve(processPhoto(image, aspect, css, portrait, blur, name, effect, watermark));
                    } catch (Throwable t) {
                        call.reject("photo: " + t);
                    } finally {
                        image.close();
                    }
                }

                @Override
                public void onError(ImageCaptureException e) {
                    call.reject("capture: " + e.getMessage());
                }
            });
        });
    }

    private JSObject processPhoto(ImageProxy image, double aspect, String css, boolean portrait,
                                  int blur, String name, String effect, boolean watermark) throws Exception {
        ByteBuffer buf = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buf.remaining()];
        buf.get(bytes);
        int rot = image.getImageInfo().getRotationDegrees();
        Bitmap bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
        bytes = null;
        if (bmp == null) throw new Exception("decode failed");

        if (rot != 0) {
            Matrix mx = new Matrix();
            mx.postRotate(rot);
            Bitmap r = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), mx, true);
            if (r != bmp) bmp.recycle();
            bmp = r;
        }
        if (funMode && fun != null) bmp = fun.process(bmp, effect, front);
        int w = bmp.getWidth(), h = bmp.getHeight(), cw = w, ch = h;
        if ((double) w / h > aspect) cw = (int) Math.round(h * aspect); else ch = (int) Math.round(w / aspect);
        if (cw != w || ch != h) {
            Bitmap c = Bitmap.createBitmap(bmp, (w - cw) / 2, (h - ch) / 2, cw, ch);
            if (c != bmp) bmp.recycle();
            bmp = c;
        }
        if (portrait) {
            Bitmap p = portraitBlur(bmp, blur);
            if (p != bmp) bmp.recycle();
            bmp = p;
        }
        ColorMatrix cm = cssToMatrix(css);
        if (cm != null) {
            Bitmap out = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
            Paint p = new Paint(Paint.FILTER_BITMAP_FLAG);
            p.setColorFilter(new ColorMatrixColorFilter(cm));
            new Canvas(out).drawBitmap(bmp, 0, 0, p);
            bmp.recycle();
            bmp = out;
        }
        if (watermark) bmp = drawWatermark(bmp);
        Uri uri = saveJpeg(bmp, name);

        int tw = 160, th = Math.max(1, bmp.getHeight() * tw / bmp.getWidth());
        Bitmap t = Bitmap.createScaledBitmap(bmp, tw, th, true);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        t.compress(Bitmap.CompressFormat.JPEG, 70, bos);
        t.recycle();
        bmp.recycle();

        JSObject o = new JSObject();
        o.put("uri", uri.toString());
        o.put("thumb", "data:image/jpeg;base64," + Base64.encodeToString(bos.toByteArray(), Base64.NO_WRAP));
        return o;
    }

    /** Белый логотип DataLaw в правом нижнем углу (ресурс dl_watermark). */
    private Bitmap drawWatermark(Bitmap bmp) {
        try {
            int id = getContext().getResources().getIdentifier("dl_watermark", "drawable", getContext().getPackageName());
            if (id == 0) return bmp;
            Bitmap wm = BitmapFactory.decodeResource(getContext().getResources(), id);
            if (wm == null) return bmp;
            Bitmap out = bmp.isMutable() ? bmp : bmp.copy(Bitmap.Config.ARGB_8888, true);
            if (out != bmp) bmp.recycle();
            int w = out.getWidth(), h = out.getHeight(), m0 = Math.min(w, h);
            int tw = Math.round(m0 * 0.30f);
            int th = Math.max(1, Math.round(tw * (float) wm.getHeight() / wm.getWidth()));
            int mg = Math.round(m0 * 0.035f);
            Paint p = new Paint(Paint.FILTER_BITMAP_FLAG);
            p.setAlpha(235);
            new Canvas(out).drawBitmap(wm, null, new Rect(w - tw - mg, h - th - mg, w - mg, h - mg), p);
            wm.recycle();
            return out;
        } catch (Throwable t) {
            Log.w(TAG, "watermark failed", t);
            return bmp;
        }
    }

    private Uri saveJpeg(Bitmap bmp, String name) throws Exception {
        ContentResolver cr = getContext().getContentResolver();
        ContentValues v = new ContentValues();
        v.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
        v.put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg");
        v.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DCIM + "/DataLaw");
        v.put(MediaStore.MediaColumns.IS_PENDING, 1);
        Uri uri = cr.insert(MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), v);
        if (uri == null) throw new Exception("MediaStore insert failed");
        try (OutputStream os = cr.openOutputStream(uri)) {
            bmp.compress(Bitmap.CompressFormat.JPEG, 95, os);
        }
        ContentValues u = new ContentValues();
        u.put(MediaStore.MediaColumns.IS_PENDING, 0);
        cr.update(uri, u, null, null);
        return uri;
    }

    /* ====================== портрет ====================== */

    private Bitmap portraitBlur(Bitmap src, int strength) throws Exception {
        final int w = src.getWidth(), h = src.getHeight();

        // 1) маска человека на уменьшенной копии
        float ms = 480f / Math.max(w, h);
        Bitmap small = Bitmap.createScaledBitmap(src, Math.max(1, Math.round(w * ms)), Math.max(1, Math.round(h * ms)), true);
        if (segmenter == null) {
            segmenter = Segmentation.getClient(new SelfieSegmenterOptions.Builder()
                .setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE).build());
        }
        SegmentationMask mask = Tasks.await(segmenter.process(InputImage.fromBitmap(small, 0)));
        small.recycle();
        final int mw = mask.getWidth(), mh = mask.getHeight();
        ByteBuffer bb = mask.getBuffer();
        bb.rewind();
        final float[] m = new float[mw * mh];
        for (int i = 0; i < m.length; i++) m[i] = bb.getFloat();

        // 2) размытая копия (уменьшить -> box blur -> растянуть)
        int bw = Math.max(8, w / 6), bh = Math.max(8, h / 6);
        Bitmap sm = Bitmap.createScaledBitmap(src, bw, bh, true);
        int[] px = new int[bw * bh], tmp = new int[bw * bh];
        sm.getPixels(px, 0, bw, 0, 0, bw, bh);
        sm.recycle();
        int r = Math.max(2, Math.round(bw / 50f * Math.max(1, Math.min(3, strength))));
        for (int i = 0; i < 2; i++) {
            blurPass(px, tmp, bw, bh, r, true);
            blurPass(tmp, px, bw, bh, r, false);
        }
        Bitmap bl = Bitmap.createBitmap(bw, bh, Bitmap.Config.ARGB_8888);
        bl.setPixels(px, 0, bw, 0, 0, bw, bh);
        Bitmap big = Bitmap.createScaledBitmap(bl, w, h, true);
        bl.recycle();

        // 3) смешивание по маске (построчно, чтобы не раздувать память)
        Bitmap out = src.copy(Bitmap.Config.ARGB_8888, true);
        int[] x0 = new int[w], x1 = new int[w];
        float[] tx = new float[w];
        for (int x = 0; x < w; x++) {
            float fx = (x + 0.5f) * mw / w - 0.5f;
            int i0 = Math.max(0, Math.min(mw - 1, (int) Math.floor(fx)));
            x0[x] = i0; x1[x] = Math.min(mw - 1, i0 + 1);
            tx[x] = Math.max(0f, Math.min(1f, fx - i0));
        }
        int[] a = new int[w], b = new int[w];
        for (int y = 0; y < h; y++) {
            out.getPixels(a, 0, w, 0, y, w, 1);
            big.getPixels(b, 0, w, 0, y, w, 1);
            float fy = (y + 0.5f) * mh / h - 0.5f;
            int y0 = Math.max(0, Math.min(mh - 1, (int) Math.floor(fy)));
            int y1 = Math.min(mh - 1, y0 + 1);
            float ty = Math.max(0f, Math.min(1f, fy - y0));
            for (int x = 0; x < w; x++) {
                float top = m[y0 * mw + x0[x]] * (1 - tx[x]) + m[y0 * mw + x1[x]] * tx[x];
                float bot = m[y1 * mw + x0[x]] * (1 - tx[x]) + m[y1 * mw + x1[x]] * tx[x];
                float conf = top * (1 - ty) + bot * ty;
                float t = Math.max(0f, Math.min(1f, (conf - 0.35f) / 0.4f));
                t = t * t * (3 - 2 * t);
                if (t >= 0.999f) continue;
                int p = a[x], q = b[x];
                float k = 1 - t;
                int rr = (int) (((p >> 16) & 255) * t + ((q >> 16) & 255) * k);
                int gg = (int) (((p >> 8) & 255) * t + ((q >> 8) & 255) * k);
                int bb2 = (int) ((p & 255) * t + (q & 255) * k);
                a[x] = 0xFF000000 | (rr << 16) | (gg << 8) | bb2;
            }
            out.setPixels(a, 0, w, 0, y, w, 1);
        }
        big.recycle();
        return out;
    }

    private static void blurPass(int[] src, int[] dst, int w, int h, int r, boolean horiz) {
        int n = horiz ? w : h, lines = horiz ? h : w, div = 2 * r + 1;
        for (int l = 0; l < lines; l++) {
            int base = horiz ? l * w : l, step = horiz ? 1 : w;
            int sr = 0, sg = 0, sb = 0;
            for (int i = -r; i <= r; i++) {
                int p = src[base + Math.max(0, Math.min(n - 1, i)) * step];
                sr += (p >> 16) & 255; sg += (p >> 8) & 255; sb += p & 255;
            }
            for (int i = 0; i < n; i++) {
                dst[base + i * step] = 0xFF000000 | ((sr / div) << 16) | ((sg / div) << 8) | (sb / div);
                int add = src[base + Math.min(n - 1, i + r + 1) * step];
                int sub = src[base + Math.max(0, i - r) * step];
                sr += ((add >> 16) & 255) - ((sub >> 16) & 255);
                sg += ((add >> 8) & 255) - ((sub >> 8) & 255);
                sb += (add & 255) - (sub & 255);
            }
        }
    }

    /* ====================== видео ====================== */

    @SuppressLint("MissingPermission")
    @PluginMethod
    public void videoStart(PluginCall call) {
        if (!funMode && (videoCapture == null || recording != null)) { call.reject("video not ready"); return; }
        final String name = call.getString("name", "DataLaw_" + System.currentTimeMillis());
        ui(() -> {
            try {
                if (funMode) {
                    fun.startRecording(name, getPermissionState("microphone") == PermissionState.GRANTED);
                    call.resolve();
                    return;
                }
                ContentValues cv = new ContentValues();
                cv.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
                cv.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DCIM + "/DataLaw");
                MediaStoreOutputOptions opts = new MediaStoreOutputOptions.Builder(
                    getContext().getContentResolver(), MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
                    .setContentValues(cv).build();
                PendingRecording pr = videoCapture.getOutput().prepareRecording(getContext(), opts);
                if (getPermissionState("microphone") == PermissionState.GRANTED) pr = pr.withAudioEnabled();
                recording = pr.start(ContextCompat.getMainExecutor(getContext()), ev -> {
                    if (ev instanceof VideoRecordEvent.Finalize) onFinalize((VideoRecordEvent.Finalize) ev);
                });
                call.resolve();
            } catch (Exception e) {
                call.reject("video: " + e.getMessage());
            }
        });
    }

    private void onFinalize(VideoRecordEvent.Finalize fin) {
        recording = null;
        final PluginCall sc = stopCall;
        stopCall = null;
        if (fin.hasError()) {
            if (sc != null) sc.reject("video error " + fin.getError());
            return;
        }
        resolveVideo(fin.getOutputResults().getOutputUri(), sc);
    }

    private void resolveVideo(final Uri uri, final PluginCall sc) {
        io.execute(() -> {
            JSObject o = new JSObject();
            o.put("uri", uri.toString());
            String thumb = "";
            try {
                MediaMetadataRetriever mmr = new MediaMetadataRetriever();
                mmr.setDataSource(getContext(), uri);
                Bitmap f = mmr.getFrameAtTime(0);
                mmr.release();
                if (f != null) {
                    Bitmap t = Bitmap.createScaledBitmap(f, 160, Math.max(1, f.getHeight() * 160 / f.getWidth()), true);
                    ByteArrayOutputStream bos = new ByteArrayOutputStream();
                    t.compress(Bitmap.CompressFormat.JPEG, 70, bos);
                    thumb = "data:image/jpeg;base64," + Base64.encodeToString(bos.toByteArray(), Base64.NO_WRAP);
                }
            } catch (Exception ignored) { }
            o.put("thumb", thumb);
            if (sc != null) sc.resolve(o);
        });
    }

    @PluginMethod
    public void videoStop(PluginCall call) {
        if (funMode && fun != null) {
            io.execute(() -> {
                try { resolveVideo(fun.stopRecording(), call); }
                catch (Exception e) { call.reject("video: " + e.getMessage()); }
            });
            return;
        }
        ui(() -> {
            if (recording == null) { call.reject("not recording"); return; }
            stopCall = call;
            recording.stop();
        });
    }

    /* ====================== прочее ====================== */

    @PluginMethod
    public void open(PluginCall call) {
        try {
            Intent i = new Intent(Intent.ACTION_VIEW);
            i.setDataAndType(Uri.parse(call.getString("uri", "")), call.getString("mime", "image/jpeg"));
            i.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_ACTIVITY_NEW_TASK);
            getContext().startActivity(i);
            call.resolve();
        } catch (Exception e) {
            call.reject(e.getMessage());
        }
    }

    @Override
    protected void handleOnPause() {
        if (funMode && fun != null && fun.isRecording()) {
            io.execute(() -> { try { fun.stopRecording(); } catch (Exception ignored) { } });
        }
        super.handleOnPause();
    }

    @Override
    protected void handleOnDestroy() {
        io.shutdown();
        if (segmenter != null) segmenter.close();
        super.handleOnDestroy();
    }
}
