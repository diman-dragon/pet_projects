use crate::db::connection::open_connection;
use crate::db::cards;

#[tauri::command]
pub fn card_delete(db_path: String, card_id: i64, deck_id: i64) -> Result<(), String> {
    let connection = open_connection(&db_path)?;
    cards::delete_card(&connection, card_id, deck_id)
}
