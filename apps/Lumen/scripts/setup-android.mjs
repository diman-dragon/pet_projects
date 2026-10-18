// Настраивает сгенерированный `cap add android` проект под Lumen.
// Идемпотентен: можно запускать сколько угодно раз.
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const A = path.join(root, 'android');
const PKG_DIR = path.join(A, 'app/src/main/java/com/lumen/gallery');
const RES = path.join(A, 'app/src/main/res');

if (!fs.existsSync(A)) {
  console.error('Папки android/ нет. Сначала: npx cap add android');
  process.exit(1);
}

const write = (p, s) => {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, s);
};

// 1. Java: плагин + MainActivity ------------------------------------------------
fs.mkdirSync(PKG_DIR, { recursive: true });
const srcDir = path.join(root, 'native/android/java/com/lumen/gallery');
for (const f of fs.readdirSync(srcDir)) fs.copyFileSync(path.join(srcDir, f), path.join(PKG_DIR, f));

write(
  path.join(PKG_DIR, 'MainActivity.java'),
  `package com.lumen.gallery;

import android.os.Bundle;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(MediaLibraryPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
`
);

// 2. Manifest: разрешения -------------------------------------------------------
const manifestPath = path.join(A, 'app/src/main/AndroidManifest.xml');
let manifest = fs.readFileSync(manifestPath, 'utf8');
if (!manifest.includes('READ_MEDIA_IMAGES')) {
  const perms = `
    <uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
    <uses-permission android:name="android.permission.READ_MEDIA_VIDEO" />
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" android:maxSdkVersion="32" />
    <uses-permission android:name="android.permission.ACCESS_MEDIA_LOCATION" />
    <uses-permission android:name="android.permission.MANAGE_MEDIA" />
`;
  manifest = manifest.replace(/<application/, `${perms}\n    <application`);
}
if (!manifest.includes('android:largeHeap')) {
  manifest = manifest.replace(/<application/, '<application android:largeHeap="true"');
}
fs.writeFileSync(manifestPath, manifest);

// 3. minSdk = 30 (scoped storage без костылей) ---------------------------------
const varsPath = path.join(A, 'variables.gradle');
if (fs.existsSync(varsPath)) {
  let v = fs.readFileSync(varsPath, 'utf8');
  v = v.replace(/minSdkVersion\s*=\s*\d+/, 'minSdkVersion = 30');
  fs.writeFileSync(varsPath, v);
}

// 4. Подпись (если заданы переменные окружения) --------------------------------
const gradlePath = path.join(A, 'app/build.gradle');
let gradle = fs.readFileSync(gradlePath, 'utf8');
if (!gradle.includes('exifinterface')) {
  gradle = gradle.replace(/dependencies\s*\{/, 'dependencies {\n    implementation "androidx.exifinterface:exifinterface:1.3.7"');
}
if (!gradle.includes('LUMEN_LINT')) {
  gradle += `
// LUMEN_LINT: lint не должен валить сборку
android {
    lint {
        abortOnError false
        checkReleaseBuilds false
    }
}
`;
}
if (!gradle.includes('LUMEN_SIGNING')) {
  gradle += `
// LUMEN_SIGNING: подпись постоянным ключом, если задан KEYSTORE_PATH (см. README)
if (System.getenv("KEYSTORE_PATH")) {
    android {
        signingConfigs {
            lumen {
                storeFile file(System.getenv("KEYSTORE_PATH"))
                storePassword System.getenv("KEYSTORE_PASSWORD")
                keyAlias System.getenv("KEY_ALIAS")
                keyPassword System.getenv("KEY_PASSWORD")
            }
        }
        buildTypes {
            debug { signingConfig signingConfigs.lumen }
            release { signingConfig signingConfigs.lumen }
        }
    }
}
`;
  fs.writeFileSync(gradlePath, gradle);
}

// 5. Иконка и сплэш -------------------------------------------------------------
write(
  path.join(RES, 'drawable/lumen_bg.xml'),
  `<?xml version="1.0" encoding="utf-8"?>
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="108dp" android:height="108dp" android:viewportWidth="108" android:viewportHeight="108">
    <path android:fillColor="#0E0E11" android:pathData="M0,0h108v108h-108z" />
</vector>
`
);
write(
  path.join(RES, 'drawable/lumen_fg.xml'),
  `<?xml version="1.0" encoding="utf-8"?>
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="108dp" android:height="108dp" android:viewportWidth="108" android:viewportHeight="108">
    <path android:strokeColor="#FF8A3D" android:strokeWidth="5" android:strokeLineJoin="round"
        android:pathData="M36,34h36a6,6 0 0 1 6,6v28a6,6 0 0 1 -6,6h-36a6,6 0 0 1 -6,-6v-28a6,6 0 0 1 6,-6z" />
    <path android:fillColor="#FF8A3D" android:pathData="M64,45m-5,0a5,5 0 1 1 10,0a5,5 0 1 1 -10,0" />
    <path android:fillColor="#FF8A3D" android:pathData="M32,68l14,-16l10,11l7,-7l15,15v2a4,4 0 0 1 -4,4h-38a4,4 0 0 1 -4,-4z" />
</vector>
`
);
const adaptive = `<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@drawable/lumen_bg" />
    <foreground android:drawable="@drawable/lumen_fg" />
</adaptive-icon>
`;
write(path.join(RES, 'mipmap-anydpi-v26/ic_launcher.xml'), adaptive);
write(path.join(RES, 'mipmap-anydpi-v26/ic_launcher_round.xml'), adaptive);

// сплэш: однотонный фон вместо логотипа Capacitor
for (const d of fs.readdirSync(RES)) {
  if (!d.startsWith('drawable')) continue;
  for (const f of fs.readdirSync(path.join(RES, d))) {
    if (/^splash\.(png|jpg|webp)$/.test(f)) fs.unlinkSync(path.join(RES, d, f));
  }
}
write(
  path.join(RES, 'drawable/splash.xml'),
  `<?xml version="1.0" encoding="utf-8"?>
<shape xmlns:android="http://schemas.android.com/apk/res/android" android:shape="rectangle">
    <solid android:color="#0B0B0D" />
</shape>
`
);

console.log('✔ Android-проект настроен для Lumen');
