use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::grammar;

#[tauri::command]
pub fn grammar_unit_list(db_path: String) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    grammar::list_grammar_units(&connection)
}
