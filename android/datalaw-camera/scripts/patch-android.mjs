// Запускается ПОСЛЕ `npx cap add android` и ДО `npx cap sync`.
// Идемпотентен: можно запускать сколько угодно раз.
//  - Java-плагины и MainActivity (пакет подставляется из capacitor.config.json)
//  - брендовые ресурсы: иконка (adaptive + themed), сплэш, водяной знак
//  - AndroidManifest: права, ориентация, тема запуска
//  - app/datalaw.gradle: подпись, versionCode/versionName, minSdk, зависимости CameraX/ML Kit
//  - www/version.js: версия и ссылки для экрана «О приложении»
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const rd = (p) => fs.readFileSync(p, 'utf8');
const cfg = JSON.parse(rd(path.join(root, 'capacitor.config.json')));
const brand = JSON.parse(rd(path.join(root, 'brand', 'brand.json')));
const pkg = JSON.parse(rd(path.join(root, 'package.json')));
const log = (m) => console.log('[patch]', m);
const fail = (m) => { console.error('[patch] ОШИБКА:', m); process.exit(1); };

if (cfg.appId !== brand.appId) fail(`appId в capacitor.config.json (${cfg.appId}) != brand/brand.json (${brand.appId})`);
if (cfg.appName !== brand.name) fail(`appName в capacitor.config.json (${cfg.appName}) != brand/brand.json (${brand.name})`);

const androidDir = path.join(root, 'android');
if (!fs.existsSync(androidDir)) fail('android/ не найден. Сначала: npx cap add android');
const main = path.join(androidDir, 'app', 'src', 'main');
const res = path.join(main, 'res');

const versionName = process.env.DL_VERSION_NAME || pkg.version;
const versionCode = parseInt(process.env.DL_VERSION_CODE || '1', 10);
const build = process.env.DL_BUILD || 'dev';

/* ---------- 0) version.js для интерфейса ---------- */
fs.writeFileSync(path.join(root, 'www', 'version.js'),
  'window.DL = ' + JSON.stringify({
    name: brand.name, short: brand.shortName, version: versionName, code: versionCode, build,
    accent: brand.colors.accent, website: brand.website, privacyUrl: brand.privacyUrl, email: brand.supportEmail,
  }, null, 2) + ';\n');
log(`version.js: ${versionName} (${versionCode})`);

/* ---------- 1) Java ---------- */
const javaDir = path.join(main, 'java', ...cfg.appId.split('.'));
fs.mkdirSync(javaDir, { recursive: true });
for (const f of fs.readdirSync(path.join(root, 'native')).filter((n) => n.endsWith('.java'))) {
  const src = rd(path.join(root, 'native', f)).replace(/^package\s+[\w.]+;/m, `package ${cfg.appId};`);
  fs.writeFileSync(path.join(javaDir, f), src);
}
// если шаблон Capacitor сгенерировал Kotlin-версию MainActivity — убираем, иначе будет дубль класса
for (const dir of ['java', 'kotlin']) {
  const kt = path.join(main, dir, ...cfg.appId.split('.'), 'MainActivity.kt');
  if (fs.existsSync(kt)) { fs.rmSync(kt); log('удалён MainActivity.kt'); }
}
log('Java: ' + fs.readdirSync(javaDir).join(', '));

/* ---------- 2) ресурсы бренда ---------- */
function copyTree(src, dst) {
  for (const e of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, e.name), d = path.join(dst, e.name);
    if (e.isDirectory()) { fs.mkdirSync(d, { recursive: true }); copyTree(s, d); } else fs.copyFileSync(s, d);
  }
}
const brandRes = path.join(root, 'android-res');
if (!fs.existsSync(path.join(brandRes, 'drawable-nodpi', 'dl_ic_foreground.png'))) {
  fail('нет android-res/ (иконки). Запустите: npm run brand');
}
copyTree(brandRes, res);
// splash.png шаблона Capacitor (во всех плотностях) заменяем сплошным цветом бренда:
// на него ссылается родительская тема, а растягиваемая картинка исказилась бы.
const solidSplash = path.join(brandRes, 'drawable', 'splash.png');
for (const d of fs.readdirSync(res).filter((n) => /^drawable/.test(n))) {
  const p = path.join(res, d, 'splash.png');
  if (fs.existsSync(p)) fs.copyFileSync(solidSplash, p);
}

const adaptive = (mono) => `<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@drawable/dl_ic_background"/>
    <foreground android:drawable="@drawable/dl_ic_foreground"/>${mono ? '\n    <monochrome android:drawable="@drawable/dl_ic_monochrome"/>' : ''}
</adaptive-icon>
`;
for (const [dir, mono] of [['mipmap-anydpi-v26', false], ['mipmap-anydpi-v33', true]]) {
  fs.mkdirSync(path.join(res, dir), { recursive: true });
  fs.writeFileSync(path.join(res, dir, 'ic_launcher.xml'), adaptive(mono));
  fs.writeFileSync(path.join(res, dir, 'ic_launcher_round.xml'), adaptive(mono));
}
fs.mkdirSync(path.join(res, 'drawable'), { recursive: true });
fs.writeFileSync(path.join(res, 'drawable', 'dl_splash_bg.xml'), `<?xml version="1.0" encoding="utf-8"?>
<layer-list xmlns:android="http://schemas.android.com/apk/res/android">
    <item android:drawable="@color/dl_splash_bg"/>
    <item android:width="168dp" android:height="168dp" android:gravity="center" android:drawable="@drawable/dl_splash_icon"/>
</layer-list>
`);
fs.mkdirSync(path.join(res, 'values'), { recursive: true });

/* ---------- 3) тема запуска (сплэш DataLaw) ---------- */
const stylesPath = path.join(res, 'values', 'styles.xml');
const styles = fs.existsSync(stylesPath) ? rd(stylesPath) : '';
const canTheme = styles.includes('AppTheme.NoActionBarLaunch') && styles.includes('Theme.SplashScreen');
const porcelain = brand.colors.porcelain;
fs.writeFileSync(path.join(res, 'values', 'dl_theme.xml'), `<?xml version="1.0" encoding="utf-8"?>
<resources>
    <color name="dl_splash_bg">${porcelain}</color>
    <color name="dl_accent">${brand.colors.accent}</color>${canTheme ? `
    <style name="AppTheme.NoActionBarLaunch.DataLaw" parent="AppTheme.NoActionBarLaunch">
        <item name="android:background">@drawable/dl_splash_bg</item>
        <item name="windowSplashScreenBackground">@color/dl_splash_bg</item>
        <item name="windowSplashScreenAnimatedIcon">@drawable/dl_splash_icon</item>
    </style>` : ''}
</resources>
`);
log(canTheme ? 'тема запуска: AppTheme.NoActionBarLaunch.DataLaw' : 'тема запуска: шаблон нестандартный — оставлена по умолчанию');

/* ---------- 4) AndroidManifest ---------- */
const manPath = path.join(main, 'AndroidManifest.xml');
let man = rd(manPath);
const addTop = (xml, key) => { if (!man.includes(key)) man = man.replace('</manifest>', `    ${xml}\n</manifest>`); };
addTop('<uses-permission android:name="android.permission.CAMERA" />', 'android.permission.CAMERA');
addTop('<uses-permission android:name="android.permission.RECORD_AUDIO" />', 'android.permission.RECORD_AUDIO');
addTop('<uses-permission android:name="android.permission.VIBRATE" />', 'android.permission.VIBRATE');
// WebView-режим: Capacitor запрашивает MODIFY_AUDIO_SETTINGS вместе с микрофоном; без декларации звук в видео отклоняется
addTop('<uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />', 'android.permission.MODIFY_AUDIO_SETTINGS');
addTop('<uses-feature android:name="android.hardware.camera" android:required="false" />', 'android.hardware.camera"');
addTop('<uses-feature android:name="android.hardware.camera.autofocus" android:required="false" />', 'android.hardware.camera.autofocus');
addTop('<uses-feature android:name="android.hardware.microphone" android:required="false" />', 'android.hardware.microphone');

/** Добавляет атрибут в первый тег <tag ...>, если его там ещё нет. */
function addAttr(tag, name, value) {
  const re = new RegExp(`<${tag}(\\s[^>]*?)?(/?)>`);
  const m = man.match(re);
  if (!m) fail(`в манифесте нет <${tag}>`);
  if (new RegExp(`${name.replace(/[.]/g, '\\.')}\\s*=`).test(m[0])) return;
  man = man.replace(re, (all, attrs = '', slash) => `<${tag}${attrs}\n        ${name}="${value}"${slash}>`);
}
addAttr('application', 'android:largeHeap', 'true');
addAttr('activity', 'android:screenOrientation', 'portrait');
// configChanges: гарантируем density (иначе смена масштаба экрана пересоздаёт Activity посреди съёмки)
man = man.replace(/android:configChanges="([^"]*)"/, (all, v) => (v.includes('density') ? all : `android:configChanges="${v}|density"`));
if (!/android:configChanges=/.test(man)) {
  addAttr('activity', 'android:configChanges', 'orientation|keyboardHidden|keyboard|screenSize|locale|smallestScreenSize|screenLayout|uiMode|navigation|density');
}
// Android 16 (targetSdk 36) игнорирует блокировку ориентации на планшетах/складных — временный opt-out
if (!man.includes('PROPERTY_COMPAT_ALLOW_RESTRICTED_RESIZABILITY')) {
  man = man.replace('</application>', '    <property android:name="android.window.PROPERTY_COMPAT_ALLOW_RESTRICTED_RESIZABILITY" android:value="true" />\n    </application>');
}
if (canTheme) man = man.replace(/@style\/AppTheme\.NoActionBarLaunch(?!\.)/g, '@style/AppTheme.NoActionBarLaunch.DataLaw');
fs.writeFileSync(manPath, man);
log('AndroidManifest.xml обновлён');

/* ---------- 5) Gradle: подпись, версии, зависимости ---------- */
const CX = '1.5.3';
fs.writeFileSync(path.join(androidDir, 'app', 'datalaw.gradle'), `// Генерируется scripts/patch-android.mjs — не редактируйте вручную.
android {
    defaultConfig {
        minSdk = 29                       // MediaStore RELATIVE_PATH, без legacy-прав на хранилище
        versionCode = (System.getenv('DL_VERSION_CODE') ?: '1').toInteger()
        versionName = System.getenv('DL_VERSION_NAME') ?: '${pkg.version}'
    }
    signingConfigs {
        datalaw {
            // Один и тот же ключ у всех сборок => APK обновляется поверх установленного.
            // CI подставляет реальный ключ из секретов; без секретов используется signing/datalaw-dev.p12
            storeFile = file(System.getenv('DL_KEYSTORE') ?: "\${rootDir}/../signing/datalaw-dev.p12")
            storePassword = System.getenv('DL_STORE_PASS') ?: 'datalaw-dev'
            keyAlias = System.getenv('DL_KEY_ALIAS') ?: 'datalaw-dev'
            keyPassword = System.getenv('DL_KEY_PASS') ?: (System.getenv('DL_STORE_PASS') ?: 'datalaw-dev')
            storeType = 'pkcs12'
        }
    }
    buildTypes {
        debug { signingConfig = signingConfigs.datalaw }
        release { signingConfig = signingConfigs.datalaw }
    }
    lint {
        abortOnError = false
        checkReleaseBuilds = false
    }
}

dependencies {
    implementation "androidx.core:core-splashscreen:1.0.1"
    implementation "androidx.annotation:annotation-experimental:1.4.1"
    implementation "androidx.camera:camera-core:${CX}"
    implementation "androidx.camera:camera-camera2:${CX}"
    implementation "androidx.camera:camera-lifecycle:${CX}"
    implementation "androidx.camera:camera-video:${CX}"
    implementation "androidx.camera:camera-view:${CX}"
    implementation "com.google.mlkit:face-detection:16.1.7"
    implementation "com.google.mlkit:segmentation-selfie:16.0.0-beta6"
}
`);
const appGradle = path.join(androidDir, 'app', 'build.gradle');
let g = rd(appGradle);
if (!g.includes("datalaw.gradle")) {
  g += "\n// DataLaw Camera: подпись, версии, CameraX/ML Kit\napply from: 'datalaw.gradle'\n";
  fs.writeFileSync(appGradle, g);
}
const gp = path.join(androidDir, 'gradle.properties');
if (fs.existsSync(gp)) {
  let p = rd(gp);
  p = /^org\.gradle\.jvmargs=.*$/m.test(p)
    ? p.replace(/^org\.gradle\.jvmargs=.*$/m, 'org.gradle.jvmargs=-Xmx3072m -Dfile.encoding=UTF-8')
    : p + '\norg.gradle.jvmargs=-Xmx3072m -Dfile.encoding=UTF-8\n';
  fs.writeFileSync(gp, p);
}
log('gradle: datalaw.gradle подключён');
log('готово');
