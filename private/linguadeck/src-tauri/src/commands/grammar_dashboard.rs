use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::grammar;

#[tauri::command]
pub fn grammar_dashboard(db_path: String) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    grammar::get_grammar_dashboard(&connection)
}
