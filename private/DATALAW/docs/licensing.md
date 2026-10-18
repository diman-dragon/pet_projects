# Licensing / Лицензирование

## English Version

### Master Key

The current master key for local/offline license management is stored in:
`.secrets/license-master.key`

This file is ignored by Git and is required for:

- Validating the developer master key inside the Tauri app.
- Generating issued license keys with `tools/generateLicense.ts`.

The actual value is stored only in the local secret file above and must not be copied into tracked source files.

### Generate a License

Run from the project root:

```powershell
pnpm exec tsx tools/generateLicense.ts --tier pro --days 365
```

Optional file output:

```powershell
pnpm exec tsx tools/generateLicense.ts --tier enterprise --days 365 --output docs/generated/license-enterprise.json
```

Supported tiers:

- `basic`
- `pro`
- `enterprise`

Expiry behavior:

- `--days 365` creates a 1-year license.
- `--days 0` creates a non-expiring issued license.

### Key Format

New issued keys use the short block format:
`DL1-XXXX-XXXX-XXXX-XXXX`

- Total of 16 characters after the `DL1-` prefix.
- The key is self-contained and includes the Tier, Expiry Date, and a 64-bit HMAC signature.
- Characters are chosen to avoid confusion (no 0, 1, I, O).
- Case-insensitive (will be normalized to uppercase).

Example:
`DL1-H8N2-K5M9-P1R4-T7X3`

### Secret Rotation

If you rotate the master key:

1. Update `.secrets/license-master.key`.
2. Rebuild the Tauri app so the new secret is embedded at build time.
3. Re-issue all activation keys, because old issued keys will no longer validate.

---

## Русская версия

### Мастер-ключ

Текущий мастер-ключ для управления локальными/офлайн лицензиями хранится в:
`.secrets/license-master.key`

Этот файл игнорируется Git и необходим для:

- Валидации мастер-ключа разработчика внутри приложения Tauri.
- Генерации лицензионных ключей с помощью `tools/generateLicense.ts`.

Значение должно храниться только в локальном секретном файле и не должно копироваться в отслеживаемые исходные файлы.

### Генерация лицензии

Запустите из корня проекта:

```powershell
pnpm exec tsx tools/generateLicense.ts --tier pro --days 365
```

Вывод в файл (опционально):

```powershell
pnpm exec tsx tools/generateLicense.ts --tier enterprise --days 365 --output docs/generated/license-enterprise.json
```

Поддерживаемые уровни (tiers):

- `basic`
- `pro`
- `enterprise`

Срок действия:

- `--days 365` создает лицензию на 1 год.
- `--days 0` создает бессрочную лицензию.

### Формат ключа

Новые ключи используют короткий блочный формат:
`DL1-XXXX-XXXX-XXXX-XXXX`

- Всего 16 символов после префикса `DL1-`.
- Ключ содержит в себе Уровень (Tier), Дату истечения и 64-битную HMAC-подпись.
- Набор символов выбран так, чтобы избежать путаницы (без 0, 1, I, O).
- Регистронезависимый (приводится к верхнему регистру).

Пример:
`DL1-H8N2-K5M9-P1R4-T7X3`

### Ротация секрета

Если вы меняете мастер-ключ:

1. Обновите `.secrets/license-master.key`.
2. Пересоберите приложение Tauri, чтобы новый секрет был встроен на этапе сборки.
3. Перевыпустите все ключи активации, так как старые ключи перестанут проходить проверку.
