use crate::db::connection::open_connection;
use crate::db::cards;
use crate::db::models::CardInput;

#[tauri::command]
pub fn card_create(db_path: String, deck_id: i64, input: CardInput) -> Result<(), String> {
    let connection = open_connection(&db_path)?;
    cards::create_card(&connection, deck_id, input)
}
