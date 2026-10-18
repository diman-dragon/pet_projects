package com.lumen.gallery;

import android.Manifest;
import android.app.Activity;
import android.app.PendingIntent;
import android.content.ContentResolver;
import android.content.ContentUris;
import android.content.ContentValues;
import android.content.Intent;
import android.database.ContentObserver;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.provider.Settings;
import android.util.Size;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.IntentSenderRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.exifinterface.media.ExifInterface;

import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.PermissionState;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;
import com.getcapacitor.annotation.PermissionCallback;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * MediaLibrary — нативный плагин Lumen.
 * Всё работает через MediaStore (scoped storage), minSdk = 30.
 */
@CapacitorPlugin(
    name = "MediaLibrary",
    permissions = {
        @Permission(alias = "media", strings = { Manifest.permission.READ_MEDIA_IMAGES, Manifest.permission.READ_MEDIA_VIDEO }),
        @Permission(alias = "storage", strings = { Manifest.permission.READ_EXTERNAL_STORAGE }),
        @Permission(alias = "location", strings = { Manifest.permission.ACCESS_MEDIA_LOCATION })
    }
)
public class MediaLibraryPlugin extends Plugin {

    /** Очередь LIFO: то, что на экране сейчас, обрабатывается первым. */
    private static final class LifoQueue extends LinkedBlockingDeque<Runnable> {
        @Override
        public boolean offer(Runnable r) {
            return offerFirst(r);
        }
    }

    private static final class Item {
        long id;
        boolean video;
        String name;
        String dir;
        String mime;
        long taken;
        long added;
        long size;
        int w;
        int h;
        long dur;
    }

    private static final class Pending {
        String type;
        PluginCall call;
        List<Uri> uris;
        String dest;
        String newName;
    }

    private volatile List<Item> items = new ArrayList<>();
    private final ExecutorService io = Executors.newSingleThreadExecutor();
    private ThreadPoolExecutor thumbPool;
    private final Set<String> inflight = ConcurrentHashMap.newKeySet();
    private File thumbDir;
    private ContentObserver observer;
    private final Handler main = new Handler(Looper.getMainLooper());
    private Runnable notifyChanged;
    private ActivityResultLauncher<IntentSenderRequest> sender;
    private Pending pending;

    // ------------------------------------------------------------------ lifecycle

    @Override
    public void load() {
        thumbDir = new File(getContext().getCacheDir(), "thumbs");
        //noinspection ResultOfMethodCallIgnored
        thumbDir.mkdirs();
        thumbPool = new ThreadPoolExecutor(3, 3, 30, TimeUnit.SECONDS, new LifoQueue());
        sender = getActivity().registerForActivityResult(
            new ActivityResultContracts.StartIntentSenderForResult(),
            r -> onSenderResult(r.getResultCode())
        );
        notifyChanged = () -> notifyListeners("changed", new JSObject());
        observer = new ContentObserver(main) {
            @Override
            public void onChange(boolean selfChange) {
                main.removeCallbacks(notifyChanged);
                main.postDelayed(notifyChanged, 1500);
            }
        };
        ContentResolver cr = getContext().getContentResolver();
        cr.registerContentObserver(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, true, observer);
        cr.registerContentObserver(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, true, observer);
    }

    @Override
    protected void handleOnDestroy() {
        try {
            getContext().getContentResolver().unregisterContentObserver(observer);
        } catch (Exception ignored) {
        }
        if (thumbPool != null) thumbPool.shutdownNow();
        io.shutdownNow();
    }

    // ------------------------------------------------------------------ permissions

    private String alias() {
        return Build.VERSION.SDK_INT >= 33 ? "media" : "storage";
    }

    private boolean granted() {
        return getPermissionState(alias()) == PermissionState.GRANTED;
    }

    private JSObject grantedResult() {
        JSObject o = new JSObject();
        o.put("granted", granted());
        return o;
    }

    @PluginMethod
    public void checkAccess(PluginCall call) {
        call.resolve(grantedResult());
    }

    @PluginMethod
    public void requestAccess(PluginCall call) {
        if (granted()) {
            afterAccess(call);
            return;
        }
        requestPermissionForAlias(alias(), call, "accessCb");
    }

    @PermissionCallback
    private void accessCb(PluginCall call) {
        afterAccess(call);
    }

    private void afterAccess(PluginCall call) {
        if (granted() && getPermissionState("location") != PermissionState.GRANTED) {
            requestPermissionForAlias("location", call, "locationCb");
        } else {
            call.resolve(grantedResult());
        }
    }

    @PermissionCallback
    private void locationCb(PluginCall call) {
        call.resolve(grantedResult());
    }

    // ------------------------------------------------------------------ scan / fetch

    private void query(Uri base, boolean video, boolean trashed, List<Item> out) {
        ContentResolver cr = getContext().getContentResolver();
        String[] proj = new String[] {
            MediaStore.MediaColumns._ID,
            MediaStore.MediaColumns.DISPLAY_NAME,
            MediaStore.MediaColumns.RELATIVE_PATH,
            MediaStore.MediaColumns.DATE_TAKEN,
            MediaStore.MediaColumns.DATE_MODIFIED,
            MediaStore.MediaColumns.DATE_ADDED,
            MediaStore.MediaColumns.SIZE,
            MediaStore.MediaColumns.WIDTH,
            MediaStore.MediaColumns.HEIGHT,
            MediaStore.MediaColumns.MIME_TYPE,
            video ? MediaStore.Video.VideoColumns.DURATION : MediaStore.Images.ImageColumns.ORIENTATION
        };
        Bundle b = new Bundle();
        b.putString(ContentResolver.QUERY_ARG_SQL_SELECTION, MediaStore.MediaColumns.SIZE + ">0");
        b.putInt(MediaStore.QUERY_ARG_MATCH_TRASHED, trashed ? MediaStore.MATCH_ONLY : MediaStore.MATCH_EXCLUDE);
        try (Cursor c = cr.query(base, proj, b, null)) {
            if (c == null) return;
            while (c.moveToNext()) {
                Item it = new Item();
                it.id = c.getLong(0);
                it.video = video;
                it.name = c.getString(1) == null ? "" : c.getString(1);
                it.dir = c.getString(2) == null ? "" : c.getString(2);
                long taken = c.getLong(3);
                if (taken <= 0) taken = c.getLong(4) * 1000L;
                it.taken = taken;
                it.added = c.getLong(5) * 1000L;
                it.size = c.getLong(6);
                it.w = c.getInt(7);
                it.h = c.getInt(8);
                it.mime = c.getString(9) == null ? "" : c.getString(9);
                if (video) {
                    it.dur = c.getLong(10);
                } else {
                    int o = c.getInt(10);
                    if (o == 90 || o == 270) {
                        int t = it.w;
                        it.w = it.h;
                        it.h = t;
                    }
                }
                out.add(it);
            }
        }
    }

    private List<Item> queryAll(boolean trashed) {
        List<Item> l = new ArrayList<>();
        query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, false, trashed, l);
        query(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, true, trashed, l);
        Collections.sort(l, (a, b) -> Long.compare(b.taken, a.taken));
        return l;
    }

    private JSONArray row(Item it) {
        JSONArray r = new JSONArray();
        r.put(it.id);
        r.put(it.video ? 1 : 0);
        r.put(it.name);
        r.put(it.dir);
        r.put(it.taken);
        r.put(it.added);
        r.put(it.size);
        r.put(it.w);
        r.put(it.h);
        r.put(it.mime);
        r.put(it.dur);
        return r;
    }

    @PluginMethod
    public void scan(PluginCall call) {
        io.execute(() -> {
            try {
                List<Item> l = queryAll(false);
                items = l;
                JSObject o = new JSObject();
                o.put("total", l.size());
                call.resolve(o);
            } catch (Exception e) {
                call.reject(String.valueOf(e.getMessage()));
            }
        });
    }

    @PluginMethod
    public void fetch(PluginCall call) {
        int off = call.getInt("offset", 0);
        int lim = call.getInt("limit", 5000);
        List<Item> l = items;
        JSONArray rows = new JSONArray();
        for (int i = off; i < l.size() && i < off + lim; i++) rows.put(row(l.get(i)));
        JSObject o = new JSObject();
        o.put("rows", rows);
        call.resolve(o);
    }

    @PluginMethod
    public void listTrash(PluginCall call) {
        io.execute(() -> {
            try {
                JSONArray rows = new JSONArray();
                for (Item it : queryAll(true)) rows.put(row(it));
                JSObject o = new JSObject();
                o.put("rows", rows);
                call.resolve(o);
            } catch (Exception e) {
                call.reject(String.valueOf(e.getMessage()));
            }
        });
    }

    // ------------------------------------------------------------------ helpers

    private Uri uri(long id, boolean video, boolean trashed) {
        return ContentUris.withAppendedId(
            video ? MediaStore.Video.Media.EXTERNAL_CONTENT_URI : MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            id
        );
    }

    private List<Uri> uris(PluginCall call, boolean trashed) {
        List<Uri> out = new ArrayList<>();
        JSArray arr = call.getArray("items");
        if (arr == null) return out;
        try {
            for (int i = 0; i < arr.length(); i++) {
                JSONObject o = arr.getJSONObject(i);
                out.add(uri(o.getLong("id"), o.getBoolean("v"), trashed));
            }
        } catch (Exception ignored) {
        }
        return out;
    }

    private Item find(long id, boolean video) {
        for (Item it : items) if (it.id == id && it.video == video) return it;
        return null;
    }

    private void ui(Runnable r) {
        getActivity().runOnUiThread(r);
    }

    // ------------------------------------------------------------------ thumbnails

    @PluginMethod
    public void queueThumbs(PluginCall call) {
        final int size = call.getInt("size", 360);
        JSArray arr = call.getArray("items");
        if (arr != null) {
            try {
                for (int i = 0; i < arr.length(); i++) {
                    JSONObject o = arr.getJSONObject(i);
                    final long id = o.getLong("id");
                    final boolean v = o.getBoolean("v");
                    final boolean t = o.optBoolean("t", false);
                    thumbPool.execute(() -> makeThumb(id, v, t, size));
                }
            } catch (Exception ignored) {
            }
        }
        call.resolve();
    }

    private void makeThumb(long id, boolean video, boolean trashed, int size) {
        final String key = (video ? "v" : "i") + id;
        final String tag = key + "_" + size;
        if (!inflight.add(tag)) return;
        String path = "";
        try {
            File f = new File(thumbDir, tag + ".webp");
            if (!f.exists() || f.length() == 0) {
                Bitmap bm = getContext().getContentResolver().loadThumbnail(uri(id, video, trashed), new Size(size, size), null);
                if (bm != null) {
                    File tmp = new File(thumbDir, tag + "." + Thread.currentThread().getId() + ".tmp");
                    try (FileOutputStream os = new FileOutputStream(tmp)) {
                        bm.compress(Bitmap.CompressFormat.WEBP_LOSSY, 82, os);
                    }
                    //noinspection ResultOfMethodCallIgnored
                    tmp.renameTo(f);
                    bm.recycle();
                }
            }
            if (f.exists() && f.length() > 0) path = f.getAbsolutePath();
        } catch (Throwable ignored) {
        } finally {
            inflight.remove(tag);
        }
        JSObject o = new JSObject();
        o.put("k", key);
        o.put("p", path);
        notifyListeners("thumb", o);
    }

    // ------------------------------------------------------------------ info

    @PluginMethod
    public void getInfo(PluginCall call) {
        final long id = call.getData().optLong("id");
        final boolean video = call.getBoolean("v", false);
        io.execute(() -> {
            JSObject o = new JSObject();
            try {
                Item it = find(id, video);
                Uri u = uri(id, video, false);
                ContentResolver cr = getContext().getContentResolver();
                if (it != null) {
                    o.put("name", it.name);
                    o.put("dir", it.dir);
                    o.put("size", it.size);
                    o.put("w", it.w);
                    o.put("h", it.h);
                    o.put("mime", it.mime);
                    o.put("taken", it.taken);
                    o.put("dur", it.dur);
                }
                try (Cursor c = cr.query(u, new String[] { MediaStore.MediaColumns.DATA }, null, null, null)) {
                    if (c != null && c.moveToFirst() && c.getString(0) != null) o.put("path", c.getString(0));
                } catch (Exception ignored) {
                }
                if (!video) {
                    Uri src = u;
                    if (getPermissionState("location") == PermissionState.GRANTED) {
                        try {
                            src = MediaStore.setRequireOriginal(u);
                        } catch (Exception ignored) {
                        }
                    }
                    try (InputStream in = cr.openInputStream(src)) {
                        if (in != null) {
                            ExifInterface ex = new ExifInterface(in);
                            putAttr(o, "make", ex.getAttribute(ExifInterface.TAG_MAKE));
                            putAttr(o, "model", ex.getAttribute(ExifInterface.TAG_MODEL));
                            putAttr(o, "exposure", ex.getAttribute(ExifInterface.TAG_EXPOSURE_TIME));
                            putAttr(o, "fnumber", ex.getAttribute(ExifInterface.TAG_F_NUMBER));
                            putAttr(o, "iso", ex.getAttribute(ExifInterface.TAG_PHOTOGRAPHIC_SENSITIVITY));
                            putAttr(o, "focal", ex.getAttribute(ExifInterface.TAG_FOCAL_LENGTH));
                            double[] ll = ex.getLatLong();
                            if (ll != null) {
                                o.put("lat", ll[0]);
                                o.put("lon", ll[1]);
                            }
                        }
                    } catch (Exception ignored) {
                    }
                }
                call.resolve(o);
            } catch (Exception e) {
                call.reject(String.valueOf(e.getMessage()));
            }
        });
    }

    private void putAttr(JSObject o, String k, String v) {
        if (v != null && !v.isEmpty()) o.put(k, v);
    }

    // ------------------------------------------------------------------ share / open

    @PluginMethod
    public void share(PluginCall call) {
        List<Uri> us = uris(call, false);
        if (us.isEmpty()) {
            call.resolve();
            return;
        }
        boolean allImg = true;
        boolean allVid = true;
        JSArray arr = call.getArray("items");
        try {
            for (int i = 0; arr != null && i < arr.length(); i++) {
                boolean v = arr.getJSONObject(i).getBoolean("v");
                if (v) allImg = false;
                else allVid = false;
            }
        } catch (Exception ignored) {
        }
        Intent it;
        if (us.size() == 1) {
            it = new Intent(Intent.ACTION_SEND);
            it.putExtra(Intent.EXTRA_STREAM, us.get(0));
        } else {
            it = new Intent(Intent.ACTION_SEND_MULTIPLE);
            it.putParcelableArrayListExtra(Intent.EXTRA_STREAM, new ArrayList<>(us));
        }
        it.setType(allImg ? "image/*" : allVid ? "video/*" : "*/*");
        it.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        final Intent chooser = Intent.createChooser(it, null);
        ui(() -> getActivity().startActivity(chooser));
        call.resolve();
    }

    @PluginMethod
    public void openWith(PluginCall call) {
        long id = call.getData().optLong("id");
        boolean v = call.getBoolean("v", false);
        Item item = find(id, v);
        Intent it = new Intent(Intent.ACTION_VIEW);
        it.setDataAndType(uri(id, v, false), item != null && !item.mime.isEmpty() ? item.mime : (v ? "video/*" : "image/*"));
        it.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        ui(() -> {
            try {
                getActivity().startActivity(Intent.createChooser(it, null));
            } catch (Exception ignored) {
            }
        });
        call.resolve();
    }

    // ------------------------------------------------------------------ manage media

    @PluginMethod
    public void manageMediaStatus(PluginCall call) {
        JSObject o = new JSObject();
        boolean supported = Build.VERSION.SDK_INT >= 31;
        o.put("supported", supported);
        o.put("granted", supported && MediaStore.canManageMedia(getContext()));
        call.resolve(o);
    }

    @PluginMethod
    public void requestManageMedia(PluginCall call) {
        if (Build.VERSION.SDK_INT >= 31) {
            Intent it = new Intent(Settings.ACTION_REQUEST_MANAGE_MEDIA, Uri.parse("package:" + getContext().getPackageName()));
            ui(() -> {
                try {
                    getActivity().startActivity(it);
                } catch (Exception ignored) {
                }
            });
        }
        call.resolve();
    }

    private boolean canManage() {
        return Build.VERSION.SDK_INT >= 31 && MediaStore.canManageMedia(getContext());
    }

    // ------------------------------------------------------------------ write operations

    @PluginMethod
    public void trash(PluginCall call) {
        start(call, "trash", uris(call, false), null, null);
    }

    @PluginMethod
    public void restore(PluginCall call) {
        start(call, "restore", uris(call, true), null, null);
    }

    @PluginMethod
    public void deleteForever(PluginCall call) {
        start(call, "delete", uris(call, true), null, null);
    }

    @PluginMethod
    public void move(PluginCall call) {
        start(call, "move", uris(call, false), call.getString("dest"), null);
    }

    @PluginMethod
    public void rename(PluginCall call) {
        start(call, "rename", uris(call, false), null, call.getString("name"));
    }

    @PluginMethod
    public void copy(PluginCall call) {
        final String dest = call.getString("dest");
        final JSArray arr = call.getArray("items");
        io.execute(() -> {
            int done = 0;
            int failed = 0;
            String err = "";
            ContentResolver cr = getContext().getContentResolver();
            try {
                for (int i = 0; arr != null && i < arr.length(); i++) {
                    JSONObject o = arr.getJSONObject(i);
                    long id = o.getLong("id");
                    boolean v = o.getBoolean("v");
                    Item src = find(id, v);
                    if (src == null) {
                        failed++;
                        continue;
                    }
                    Uri target = null;
                    try {
                        ContentValues cv = new ContentValues();
                        cv.put(MediaStore.MediaColumns.DISPLAY_NAME, src.name);
                        cv.put(MediaStore.MediaColumns.MIME_TYPE, src.mime);
                        cv.put(MediaStore.MediaColumns.RELATIVE_PATH, dest);
                        cv.put(MediaStore.MediaColumns.IS_PENDING, 1);
                        if (src.taken > 0) cv.put(MediaStore.MediaColumns.DATE_TAKEN, src.taken);
                        target = cr.insert(v ? MediaStore.Video.Media.EXTERNAL_CONTENT_URI : MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv);
                        if (target == null) throw new IllegalStateException("insert failed");
                        try (InputStream in = cr.openInputStream(uri(id, v, false)); OutputStream out = cr.openOutputStream(target)) {
                            if (in == null || out == null) throw new IllegalStateException("stream");
                            byte[] buf = new byte[1 << 16];
                            int n;
                            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
                        }
                        ContentValues fin = new ContentValues();
                        fin.put(MediaStore.MediaColumns.IS_PENDING, 0);
                        cr.update(target, fin, null, null);
                        done++;
                    } catch (Exception e) {
                        failed++;
                        err = String.valueOf(e.getMessage());
                        if (target != null) {
                            try {
                                cr.delete(target, null, null);
                            } catch (Exception ignored) {
                            }
                        }
                    }
                }
            } catch (Exception e) {
                err = String.valueOf(e.getMessage());
            }
            JSObject res = new JSObject();
            res.put("done", done);
            res.put("failed", failed);
            res.put("error", err);
            call.resolve(res);
        });
    }

    private void start(PluginCall call, String type, List<Uri> us, String dest, String name) {
        if (us.isEmpty()) {
            call.reject("empty");
            return;
        }
        Pending p = new Pending();
        p.type = type;
        p.call = call;
        p.uris = us;
        p.dest = dest;
        p.newName = name;

        // Android 12+: если у приложения есть MANAGE_MEDIA, системные диалоги не нужны (кроме удаления навсегда)
        if (canManage() && !type.equals("delete")) {
            io.execute(() -> perform(p, true));
            return;
        }
        ContentResolver cr = getContext().getContentResolver();
        PendingIntent pi;
        switch (type) {
            case "trash":
                pi = MediaStore.createTrashRequest(cr, us, true);
                break;
            case "restore":
                pi = MediaStore.createTrashRequest(cr, us, false);
                break;
            case "delete":
                pi = MediaStore.createDeleteRequest(cr, us);
                break;
            default:
                pi = MediaStore.createWriteRequest(cr, us);
        }
        pending = p;
        final PendingIntent fpi = pi;
        ui(() -> {
            try {
                sender.launch(new IntentSenderRequest.Builder(fpi.getIntentSender()).build());
            } catch (Exception e) {
                pending = null;
                call.reject(String.valueOf(e.getMessage()));
            }
        });
    }

    private void onSenderResult(int resultCode) {
        final Pending p = pending;
        pending = null;
        if (p == null) return;
        if (resultCode != Activity.RESULT_OK) {
            p.call.reject("cancelled", "cancelled");
            return;
        }
        io.execute(() -> perform(p, false));
    }

    /** direct=true — делаем сами через update(); false — система уже выполнила trash/restore/delete. */
    private void perform(Pending p, boolean direct) {
        ContentResolver cr = getContext().getContentResolver();
        int done = 0;
        int failed = 0;
        String err = "";
        for (Uri u : p.uris) {
            try {
                ContentValues cv = new ContentValues();
                switch (p.type) {
                    case "trash":
                    case "restore":
                        if (!direct) {
                            done++;
                            continue;
                        }
                        cv.put(MediaStore.MediaColumns.IS_TRASHED, p.type.equals("trash") ? 1 : 0);
                        break;
                    case "delete":
                        done++;
                        continue;
                    case "move":
                        cv.put(MediaStore.MediaColumns.RELATIVE_PATH, p.dest);
                        break;
                    case "rename":
                        cv.put(MediaStore.MediaColumns.DISPLAY_NAME, p.newName);
                        break;
                    default:
                        break;
                }
                if (cr.update(u, cv, null, null) > 0) done++;
                else failed++;
            } catch (Exception e) {
                failed++;
                err = String.valueOf(e.getMessage());
            }
        }
        JSObject res = new JSObject();
        res.put("done", done);
        res.put("failed", failed);
        res.put("error", err);
        p.call.resolve(res);
    }
}
