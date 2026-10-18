use crate::db::connection::open_connection;
use crate::db::cards::load_card;
use serde_json::Value;

#[tauri::command]
pub fn card_get(db_path: String, card_id: i64) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    load_card(&connection, card_id)
}
