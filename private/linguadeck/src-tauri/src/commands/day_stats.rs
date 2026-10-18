use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::study;

#[tauri::command]
pub fn day_stats(db_path: String, date: String) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    study::get_day_stats(&connection, &date)
}
