use crate::db::connection::open_connection;
use crate::db::logs;

#[tauri::command]
pub fn log_clear(db_path: String) -> Result<(), String> {
    let connection = open_connection(&db_path)?;
    logs::clear_logs(&connection)
}
