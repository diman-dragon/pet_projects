use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::settings::{load_settings, settings_value};

#[tauri::command]
pub fn settings_get(db_path: String) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    let settings = load_settings(&connection)?;
    Ok(settings_value(&settings))
}
