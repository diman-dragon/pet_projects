use serde_json::Value;
use crate::db::connection::open_connection;
use crate::db::grammar;
use crate::db::models::GrammarExerciseInput;

#[tauri::command]
pub fn grammar_exercise_save(db_path: String, input: GrammarExerciseInput) -> Result<Value, String> {
    let connection = open_connection(&db_path)?;
    grammar::save_grammar_exercise(&connection, input)
}
