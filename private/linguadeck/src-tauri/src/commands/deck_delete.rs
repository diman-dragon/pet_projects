use crate::db::connection::open_connection;
use crate::db::decks;

#[tauri::command]
pub fn deck_delete(db_path: String, deck_id: i64) -> Result<(), String> {
    let connection = open_connection(&db_path)?;
    decks::delete_deck(&connection, deck_id)
}
