use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::study;

#[tauri::command]
pub fn activity_history(db_path: String, days: i64) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    study::get_activity_history(&connection, days)
}
