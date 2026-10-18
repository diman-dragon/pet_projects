use crate::db::connection::open_connection;
use crate::db::cards;

#[tauri::command]
pub fn card_delete_many(db_path: String, card_ids: Vec<i64>, deck_id: i64) -> Result<(), String> {
    let connection = open_connection(&db_path)?;
    cards::delete_cards_many(&connection, card_ids, deck_id)
}
