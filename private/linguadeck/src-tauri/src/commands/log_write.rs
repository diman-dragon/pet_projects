use crate::db::connection::open_connection;
use crate::db::logs;

#[tauri::command]
pub fn log_write(
    db_path: String,
    level: String,
    scope: String,
    message: String,
    details: Option<String>,
) -> Result<(), String> {
    let connection = open_connection(&db_path)?;
    logs::write_log(&connection, &level, &scope, &message, details)
}
