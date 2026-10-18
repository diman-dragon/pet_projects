use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::grammar;

#[tauri::command]
pub fn grammar_check_unit(db_path: String, title: String) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    grammar::check_grammar_unit(&connection, &title)
}
