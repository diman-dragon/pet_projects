use rusqlite::Connection;
use chrono::Local;

pub fn open_connection(db_path: &str) -> Result<Connection, String> {
    let connection = Connection::open(db_path).map_err(|err| err.to_string())?;
    connection
        .execute("PRAGMA foreign_keys = ON;", [])
        .map_err(|err| err.to_string())?;
    Ok(connection)
}

pub fn now_string() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}
