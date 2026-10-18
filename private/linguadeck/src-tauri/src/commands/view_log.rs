use crate::db::connection::open_connection;
use crate::db::logs;

#[tauri::command]
pub fn view_log(
    db_path: String,
    card_id: i64,
    deck_id: i64,
    view_type: String,
    time_spent_ms: Option<i64>,
) -> Result<(), String> {
    let connection = open_connection(&db_path)?;
    logs::log_view(&connection, card_id, deck_id, &view_type, time_spent_ms)
}
