use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::grammar;
use crate::db::models::GrammarUnitInput;

#[tauri::command]
pub fn grammar_unit_create(db_path: String, input: GrammarUnitInput) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    grammar::create_grammar_unit(&connection, input)
}
