use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::study;

#[tauri::command]
pub fn streak(db_path: String) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    study::get_streak(&connection)
}
