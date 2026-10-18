use rusqlite::{params, Connection};
use serde_json::{json, Value};

pub fn write_log(
    connection: &Connection,
    level: &str,
    scope: &str,
    message: &str,
    details: Option<String>,
) -> Result<(), String> {
    connection
        .execute(
            "INSERT INTO app_logs (level, scope, message, details) VALUES (?1, ?2, ?3, ?4)",
            params![level, scope, message, details],
        )
        .map_err(|err| err.to_string())?;
    Ok(())
}

pub fn list_logs(connection: &Connection, limit: i64) -> Result<Value, String> {
    let mut stmt = connection
        .prepare("SELECT id, level, scope, message, details, created_at FROM app_logs ORDER BY datetime(created_at) DESC, id DESC LIMIT ?1")
        .map_err(|err| err.to_string())?;
    let rows = stmt
        .query_map([limit], |row| {
            Ok(json!({
                "id": row.get::<_, i64>(0)?,
                "level": row.get::<_, String>(1)?,
                "scope": row.get::<_, String>(2)?,
                "message": row.get::<_, String>(3)?,
                "details": row.get::<_, Option<String>>(4)?,
                "created_at": row.get::<_, String>(5)?,
            }))
        })
        .map_err(|err| err.to_string())?;
    Ok(Value::Array(
        rows.collect::<Result<Vec<_>, _>>().map_err(|err| err.to_string())?,
    ))
}

pub fn clear_logs(connection: &Connection) -> Result<(), String> {
    connection
        .execute("DELETE FROM app_logs", [])
        .map_err(|err| err.to_string())?;
    Ok(())
}

pub fn log_view(
    connection: &Connection,
    card_id: i64,
    deck_id: i64,
    view_type: &str,
    time_spent_ms: Option<i64>,
) -> Result<(), String> {
    let ms = time_spent_ms.unwrap_or(0).max(0);
    connection
        .execute(
            "INSERT INTO view_logs (card_id, deck_id, view_type, time_spent_ms, viewed_at) VALUES (?1, ?2, ?3, ?4, datetime('now'))",
            params![card_id, deck_id, view_type, ms],
        )
        .map_err(|err| err.to_string())?;
    Ok(())
}
