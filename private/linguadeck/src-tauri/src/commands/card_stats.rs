use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::cards;

#[tauri::command]
pub fn card_stats(db_path: String, card_id: i64) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    cards::get_card_stats(&connection, card_id)
}
