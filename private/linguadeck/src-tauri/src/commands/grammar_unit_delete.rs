use crate::db::connection::open_connection;
use crate::db::grammar;

#[tauri::command]
pub fn grammar_unit_delete(db_path: String, unit_id: i64) -> Result<(), String> {
    let connection = open_connection(&db_path)?;
    grammar::delete_grammar_unit(&connection, unit_id)
}
