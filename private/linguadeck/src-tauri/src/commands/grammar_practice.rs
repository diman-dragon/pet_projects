use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::grammar;

#[tauri::command]
pub fn grammar_practice(db_path: String, unit_id: i64, limit: Option<i64>) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    grammar::get_grammar_practice(&connection, unit_id, limit)
}
