package com.datalaw.folio;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.DocumentsContract;
import androidx.activity.result.ActivityResult;
import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.ActivityCallback;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/** Library folder access through the Storage Access Framework: pick a tree, list files, stage a copy for reading. */
@CapacitorPlugin(name = "Folder")
public class FolderPlugin extends Plugin {
    private static final int MAX_FILES = 5000;
    private static final int MAX_DEPTH = 8;

    @PluginMethod
    public void pick(PluginCall call) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
                | Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
        startActivityForResult(call, intent, "pickResult");
    }

    @ActivityCallback
    private void pickResult(PluginCall call, ActivityResult result) {
        if (call == null) return;
        Intent data = result.getData();
        if (result.getResultCode() != Activity.RESULT_OK || data == null || data.getData() == null) {
            call.resolve(new JSObject());
            return;
        }
        Uri tree = data.getData();
        try {
            getContext().getContentResolver().takePersistableUriPermission(tree, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (Exception ignored) {
            // permission is still valid for this session
        }
        JSObject out = new JSObject();
        out.put("uri", tree.toString());
        out.put("name", folderName(tree));
        call.resolve(out);
    }

    @PluginMethod
    public void list(final PluginCall call) {
        final String uriStr = call.getString("uri");
        if (uriStr == null) {
            call.reject("uri is required");
            return;
        }
        final Set<String> exts = new HashSet<>();
        JSArray arr = call.getArray("exts");
        if (arr != null) {
            for (int i = 0; i < arr.length(); i++) {
                String e = arr.optString(i, "");
                if (!e.isEmpty()) exts.add(e.toLowerCase(Locale.ROOT));
            }
        }
        new Thread(() -> {
            try {
                Uri tree = Uri.parse(uriStr);
                JSArray files = new JSArray();
                walk(getContext().getContentResolver(), tree, DocumentsContract.getTreeDocumentId(tree), exts, files, 0);
                JSObject out = new JSObject();
                out.put("files", files);
                call.resolve(out);
            } catch (Exception e) {
                call.reject("Cannot read folder: " + e.getMessage());
            }
        }).start();
    }

    /** Copies one document into the app cache and returns its absolute path (fallback when content:// cannot be fetched). */
    @PluginMethod
    public void stage(final PluginCall call) {
        final String uriStr = call.getString("uri");
        if (uriStr == null) {
            call.reject("uri is required");
            return;
        }
        new Thread(() -> {
            File dir = new File(getContext().getCacheDir(), "folio-stage");
            File out = new File(dir, Integer.toHexString(uriStr.hashCode()) + ".bin");
            File[] old = dir.listFiles();
            if (old != null) {
                long cutoff = System.currentTimeMillis() - 60L * 60L * 1000L;
                for (File f : old) if (f.lastModified() < cutoff) f.delete(); // staged copies are read right away; do not let the cache grow
            }
            try (InputStream in = getContext().getContentResolver().openInputStream(Uri.parse(uriStr))) {
                if (in == null) throw new IllegalStateException("cannot open");
                dir.mkdirs();
                try (OutputStream os = new FileOutputStream(out)) {
                    byte[] buf = new byte[65536];
                    int n;
                    while ((n = in.read(buf)) > 0) os.write(buf, 0, n);
                }
                JSObject res = new JSObject();
                res.put("path", out.getAbsolutePath());
                call.resolve(res);
            } catch (Exception e) {
                call.reject("Cannot read file: " + e.getMessage());
            }
        }).start();
    }

    private void walk(ContentResolver cr, Uri tree, String parentId, Set<String> exts, JSArray out, int depth) {
        Uri children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, parentId);
        String[] cols = {
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE,
            DocumentsContract.Document.COLUMN_SIZE,
            DocumentsContract.Document.COLUMN_LAST_MODIFIED
        };
        try (Cursor c = cr.query(children, cols, null, null, null)) {
            if (c == null) return;
            while (c.moveToNext() && out.length() < MAX_FILES) {
                String id = c.getString(0);
                String name = c.getString(1);
                String mime = c.getString(2);
                if (name == null || id == null) continue;
                if (DocumentsContract.Document.MIME_TYPE_DIR.equals(mime)) {
                    if (depth < MAX_DEPTH) walk(cr, tree, id, exts, out, depth + 1);
                    continue;
                }
                String lower = name.toLowerCase(Locale.ROOT);
                boolean wanted = exts.isEmpty();
                for (String e : exts) {
                    if (lower.endsWith("." + e)) { wanted = true; break; } // also compound extensions such as fb2.zip
                }
                if (!wanted) continue;
                JSObject f = new JSObject();
                f.put("uri", DocumentsContract.buildDocumentUriUsingTree(tree, id).toString());
                f.put("name", name);
                f.put("size", c.isNull(3) ? 0L : c.getLong(3));
                f.put("mtime", c.isNull(4) ? 0L : c.getLong(4));
                out.put(f);
            }
        }
    }

    private String folderName(Uri tree) {
        Uri doc = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree));
        try (Cursor c = getContext().getContentResolver().query(doc, new String[] {DocumentsContract.Document.COLUMN_DISPLAY_NAME}, null, null, null)) {
            if (c != null && c.moveToFirst() && c.getString(0) != null) return c.getString(0);
        } catch (Exception ignored) {
            // fall through to the raw id
        }
        return DocumentsContract.getTreeDocumentId(tree);
    }
}
