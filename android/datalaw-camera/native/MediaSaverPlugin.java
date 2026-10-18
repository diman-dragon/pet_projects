package com.datalaw.camera;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Потоковая запись фото/видео в галерею (MediaStore, DCIM/DataLaw).
 * JS: begin({name,mime}) -> {id}; append({id,data(base64)}); finish({id}) -> {uri}; abort({id}).
 */
@CapacitorPlugin(name = "MediaSaver")
public class MediaSaverPlugin extends Plugin {
    private final Map<String, Uri> uris = new HashMap<>();
    private final Map<String, OutputStream> streams = new HashMap<>();

    @PluginMethod
    public void begin(PluginCall call) {
        try {
            String name = call.getString("name");
            String mime = call.getString("mime", "image/jpeg");
            if (name == null) { call.reject("name required"); return; }
            boolean video = mime.startsWith("video");

            ContentResolver cr = getContext().getContentResolver();
            ContentValues v = new ContentValues();
            v.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
            v.put(MediaStore.MediaColumns.MIME_TYPE, mime);
            v.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DCIM + "/DataLaw");
            v.put(MediaStore.MediaColumns.IS_PENDING, 1);

            Uri collection = video
                ? MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
                : MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY);

            Uri uri = cr.insert(collection, v);
            if (uri == null) { call.reject("MediaStore insert failed"); return; }
            OutputStream os = cr.openOutputStream(uri);
            if (os == null) { call.reject("openOutputStream failed"); return; }

            String id = UUID.randomUUID().toString();
            synchronized (this) { uris.put(id, uri); streams.put(id, os); }
            JSObject ret = new JSObject();
            ret.put("id", id);
            call.resolve(ret);
        } catch (Exception e) {
            call.reject(e.getMessage(), e);
        }
    }

    @PluginMethod
    public void append(PluginCall call) {
        try {
            OutputStream os;
            synchronized (this) { os = streams.get(call.getString("id")); }
            String data = call.getString("data");
            if (os == null || data == null) { call.reject("bad id/data"); return; }
            os.write(Base64.decode(data, Base64.DEFAULT));
            call.resolve();
        } catch (Exception e) {
            call.reject(e.getMessage(), e);
        }
    }

    @PluginMethod
    public void finish(PluginCall call) {
        try {
            String id = call.getString("id");
            OutputStream os; Uri uri;
            synchronized (this) { os = streams.remove(id); uri = uris.remove(id); }
            if (os == null || uri == null) { call.reject("bad id"); return; }
            os.flush();
            os.close();
            ContentValues u = new ContentValues();
            u.put(MediaStore.MediaColumns.IS_PENDING, 0);
            getContext().getContentResolver().update(uri, u, null, null);
            JSObject ret = new JSObject();
            ret.put("uri", uri.toString());
            call.resolve(ret);
        } catch (Exception e) {
            call.reject(e.getMessage(), e);
        }
    }

    @PluginMethod
    public void abort(PluginCall call) {
        try {
            String id = call.getString("id");
            OutputStream os; Uri uri;
            synchronized (this) { os = streams.remove(id); uri = uris.remove(id); }
            if (os != null) { try { os.close(); } catch (Exception ignored) {} }
            if (uri != null) getContext().getContentResolver().delete(uri, null, null);
            call.resolve();
        } catch (Exception e) {
            call.reject(e.getMessage(), e);
        }
    }

    /** Включить/выключить перехват кнопок громкости как спуска затвора. */
    @PluginMethod
    public void volumeKeys(PluginCall call) {
        MainActivity.volumeCapture = Boolean.TRUE.equals(call.getBoolean("enabled", true));
        call.resolve();
    }
}
