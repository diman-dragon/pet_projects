use crate::db::connection::open_connection;
use crate::db::decks;

#[tauri::command]
pub fn deck_update(
    db_path: String,
    deck_id: i64,
    name: String,
    description: String,
    icon: String,
    color: String,
    language: Option<String>,
) -> Result<(), String> {
    let connection = open_connection(&db_path)?;
    decks::update_deck(&connection, deck_id, &name, &description, &icon, &color, language)
}
