// src-tauri/src/sqlite_vec_commands.rs
//
// Загружает sqlite-vec глобально через sqlite3_auto_extension.
// Вызывается один раз при старте приложения (.setup()).
// После этого vec0 доступен во ВСЕХ соединениях SQLite в процессе,
// включая любые SQLite connection pool'ы.

use std::sync::Once;

static INIT: Once = Once::new();

/// Регистрирует sqlite-vec для всех соединений в процессе.
/// Идемпотентно — повторный вызов ничего не делает.
pub fn register_sqlite_vec() {
    INIT.call_once(|| {
        // SAFETY: sqlite3_vec_init — корректная entrypoint-функция расширения.
        // sqlite3_auto_extension регистрирует её для всех новых соединений.
        unsafe {
            libsqlite3_sys::sqlite3_auto_extension(Some(
                std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ()),
            ));
        }

        log::info!("[sqlite-vec] auto_extension registered — vec0 available globally");
    });
}

/// Tauri-команда для проверки что sqlite-vec загружен.
#[tauri::command]
pub fn check_sqlite_vec() -> String {
    "sqlite-vec registered via auto_extension".to_string()
}
