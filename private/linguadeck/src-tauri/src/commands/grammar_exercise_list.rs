use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::grammar;

#[tauri::command]
pub fn grammar_exercise_list(db_path: String, unit_id: i64) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    grammar::list_grammar_exercises(&connection, unit_id)
}
