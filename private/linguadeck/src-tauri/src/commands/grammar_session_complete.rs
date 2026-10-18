use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::grammar;

#[tauri::command]
pub fn grammar_session_complete(
    db_path: String,
    unit_id: i64,
    correct_count: i64,
    total_count: i64,
) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    grammar::complete_grammar_session(&connection, unit_id, correct_count, total_count)
}
