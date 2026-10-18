use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::cards;

#[tauri::command]
pub fn card_list(db_path: String, deck_id: i64) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    cards::list_cards(&connection, deck_id)
}
