use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::logs;

#[tauri::command]
pub fn log_list(db_path: String, limit: i64) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    logs::list_logs(&connection, limit)
}
