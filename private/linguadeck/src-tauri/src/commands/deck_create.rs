use crate::db::connection::open_connection;
use crate::db::decks;

#[tauri::command]
pub fn deck_create(
    db_path: String,
    course_id: i64,
    name: String,
    description: String,
    icon: String,
    color: String,
    language: Option<String>,
) -> Result<(), String> {
    let connection = open_connection(&db_path)?;
    decks::create_deck(&connection, course_id, &name, &description, &icon, &color, language)
}
