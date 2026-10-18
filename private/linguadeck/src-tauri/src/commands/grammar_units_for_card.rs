use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::grammar;

#[tauri::command]
pub fn grammar_units_for_card(db_path: String, card_id: i64) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    grammar::get_grammar_units_for_card(&connection, card_id)
}
