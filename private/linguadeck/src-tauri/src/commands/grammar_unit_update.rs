use crate::db::connection::open_connection;
use crate::db::grammar;
use crate::db::models::GrammarUnitInput;

#[tauri::command]
pub fn grammar_unit_update(db_path: String, unit_id: i64, input: GrammarUnitInput) -> Result<(), String> {
    let connection = open_connection(&db_path)?;
    grammar::update_grammar_unit(&connection, unit_id, input)
}
