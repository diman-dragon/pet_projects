# DataLaw Camera — чек-лист публикации

## 0. Один раз (≈ 20 мин)
1. **Бренд-данные.** `brand/brand.json`: `supportEmail`, `website`, `privacyUrl`. `appId` (`com.datalaw.camera`) —
   подумайте СЕЙЧАС: после первой публикации он **не меняется никогда**.
2. **Ключ подписи (это и есть «чтобы обновлялось»).**
   `npm run keystore` → скопируйте 3 секрета в GitHub (Settings → Secrets → Actions):
   `DL_KEYSTORE_BASE64`, `DL_KEYSTORE_PASSWORD`, `DL_KEY_ALIAS`. Сохраните `signing/datalaw-release.p12` и пароль в надёжном месте.
3. **Политика конфиденциальности.** `npm run docs` → закоммитьте `docs/` → GitHub Settings → Pages → `main` / `/docs`.
   Полученный адрес `https://<org>.github.io/<repo>/privacy.html` впишите в `brand.json → privacyUrl` и в Play Console.
4. **Аккаунт Google Play Developer** (разовый взнос $25) и создание приложения: Play Console → «Создать приложение»
   (название «DataLaw Camera», язык RU, «Приложение», «Бесплатное»).
5. **Play App Signing:** при первой загрузке AAB согласитесь на подпись Google — ваш `datalaw-release.p12` станет *upload-ключом*.

## 1. Заполнить карточку (Play Console)
- Тексты: `store/listing_ru.md`, `store/listing_en.md`
- Иконка 512×512: `store/play-icon-512.png` · Feature graphic: `store/play-feature-graphic-1024x500.png`
- **Скриншоты** (минимум 2, лучше 4–8, 9:16): снимите на телефоне — основной экран, PRO, портрет, эффекты.
- Раздел «Безопасность данных»: `store/data-safety.md` · Возрастной рейтинг (IARC) · Целевая аудитория: 13+/все.
- Объявление о разрешениях камеры/микрофона: основная функция приложения.

## 2. Первый релиз
1. Версия: `package.json → version` (например `1.0.0`).
2. `git tag v1.0.0 && git push --tags` → Actions соберёт **APK + AAB** (versionCode растёт автоматически).
3. Скачайте артефакт `DataLaw-Camera-1.0.0-….aab` → Play Console → Тестирование → **Внутреннее** → создать релиз → загрузить AAB.
   (Первую загрузку Google требует сделать вручную.)
4. Проверьте на своём телефоне через ссылку внутреннего теста → затем «Продвинуть» в Production.

## 3. Последующие обновления — автоматически
1. Поднять `version` в `package.json`, закоммитить, поставить тег `vX.Y.Z`.
2. Если задан секрет `PLAY_SERVICE_ACCOUNT_JSON` (Play Console → Настройки → Доступ к API → сервисный аккаунт с правом
   «Релизы»), AAB сам уйдёт на internal track. Иначе — загрузите AAB вручную.
3. versionCode = 1000 + номер запуска Actions — всегда растёт; versionName = тег.

## 4. «Обновляется, а не ставится новым приложением» — три условия
| Условие | Где обеспечено |
|---|---|
| тот же `applicationId` | `capacitor.config.json` + `brand.json` (проверяет `check-brand.mjs`) |
| та же подпись | один ключ: `DL_KEYSTORE_*` (или dev-ключ в репозитории) → в Summary каждого запуска печатается SHA-256 сертификата |
| больший `versionCode` | `1000 + GITHUB_RUN_NUMBER` |
⚠ Переход с dev-ключа на реальный (и между APK из GitHub и установкой из Google Play) системой считается «другим приложением» —
один раз придётся удалить старую копию. Поэтому создайте реальный ключ ДО первой раздачи.

## 5. Требования Google Play, которые уже учтены
- targetSdk **36** (Capacitor 8) — требуется с 31.08.2026
- AAB, 64-bit, нативные библиотеки ML Kit с поддержкой страниц 16 КБ
- edge-to-edge/Android 16: обработано (иммерсивный режим, `SystemBars`, opt-out для блокировки ориентации)
