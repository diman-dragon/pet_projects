use crate::db::connection::open_connection;
use crate::db::cards;
use crate::db::models::CardInput;

#[tauri::command]
pub fn card_create_batch(
    db_path: String,
    deck_id: i64,
    inputs: Vec<CardInput>,
) -> Result<i64, String> {
    let connection = open_connection(&db_path)?;
    cards::create_card_batch(&connection, deck_id, inputs)
}
