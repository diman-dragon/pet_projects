use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::decks;

#[tauri::command]
pub fn deck_list(db_path: String) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    decks::list_decks(&connection)
}
