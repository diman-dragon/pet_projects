-- tools/license_server_schema.sql
--
-- Схема БД для серверной системы лицензирования Dorlane.
-- Используется когда LICENSE_ACTIVATION_MODE = 'server'.
-- Совместима с PostgreSQL и SQLite (для разработки).
--
-- Развернуть: psql -U postgres -d Dorlane_licenses -f license_server_schema.sql

-- ─── Таблица лицензионных ключей ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS licenses (
    id            SERIAL PRIMARY KEY,
    key           TEXT        NOT NULL UNIQUE,
    tier          TEXT        NOT NULL CHECK (tier IN ('basic', 'pro', 'enterprise')),

    -- NULL = ключ не привязан к машине до первой активации
    fingerprint   TEXT,

    -- Количество допустимых активаций (NULL = неограниченно)
    max_seats     INTEGER     DEFAULT 1,
    used_seats    INTEGER     NOT NULL DEFAULT 0,

    -- Срок действия (NULL = бессрочно)
    expires_at    TIMESTAMPTZ,
    issued_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Мета
    customer_name TEXT,
    customer_email TEXT,
    notes         TEXT,

    -- Статус
    is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
    revoked_at    TIMESTAMPTZ,
    revoked_by    TEXT
);

-- ─── Таблица активаций ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS activations (
    id            SERIAL PRIMARY KEY,
    license_id    INTEGER     NOT NULL REFERENCES licenses(id) ON DELETE CASCADE,
    fingerprint   TEXT        NOT NULL,
    activated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ip_address    TEXT,
    user_agent    TEXT,
    -- Деактивация (освобождает seat)
    deactivated_at TIMESTAMPTZ,
    UNIQUE (license_id, fingerprint)
);

-- ─── Индексы ─────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_licenses_key         ON licenses(key);
CREATE INDEX IF NOT EXISTS idx_licenses_fingerprint ON licenses(fingerprint);
CREATE INDEX IF NOT EXISTS idx_activations_license  ON activations(license_id);
CREATE INDEX IF NOT EXISTS idx_activations_fp       ON activations(fingerprint);

-- ─── Вспомогательные запросы (примеры) ───────────────────────────────────────

-- Выдать новый ключ:
-- INSERT INTO licenses (key, tier, expires_at, customer_name, customer_email)
-- VALUES ('Dorlane-PRO-A1B2-C3D4-E5F6', 'pro', NOW() + INTERVAL '1 year', 'ООО Ромашка', 'it@romashka.ru');

-- Проверить ключ при активации:
-- SELECT * FROM licenses
-- WHERE key = $1
--   AND is_active = TRUE
--   AND (expires_at IS NULL OR expires_at > NOW())
--   AND (max_seats IS NULL OR used_seats < max_seats);

-- Отозвать ключ:
-- UPDATE licenses SET is_active = FALSE, revoked_at = NOW(), revoked_by = 'admin'
-- WHERE key = $1;
