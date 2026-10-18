use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::study;

#[tauri::command]
pub fn study_session_cards(db_path: String, deck_id: i64) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    study::get_study_session_cards(&connection, deck_id)
}
