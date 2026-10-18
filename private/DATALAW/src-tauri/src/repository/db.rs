use std::{fs, path::PathBuf};

use rusqlite::{Connection, OptionalExtension};
use serde::Deserialize;
use tauri::{AppHandle, Manager};

use crate::error::{AppError, AppResult};

#[derive(Debug, Deserialize)]
struct AppConfig {
    #[serde(rename = "workDir")]
    work_dir: Option<String>,
}

pub fn open_clients_db(app: &AppHandle) -> AppResult<Connection> {
    let conn = Connection::open(clients_db_path(app)?).map_err(AppError::from)?;
    ensure_cases_counterparty_column(&conn)?;
    ensure_appeals_incoming_number_column(&conn)?;
    Ok(conn)
}

fn ensure_cases_counterparty_column(conn: &Connection) -> AppResult<()> {
    if !table_exists(conn, "cases")? {
        return Ok(());
    }
    if !has_column(conn, "cases", "counterparty")? {
        conn.execute("ALTER TABLE cases ADD COLUMN counterparty TEXT;", [])?;
    }
    if !has_column(conn, "cases", "litigation_subtype")? {
        conn.execute("ALTER TABLE cases ADD COLUMN litigation_subtype TEXT;", [])?;
    }
    Ok(())
}

fn ensure_appeals_incoming_number_column(conn: &Connection) -> AppResult<()> {
    if !table_exists(conn, "appeals")? {
        return Ok(());
    }
    if !has_column(conn, "appeals", "incoming_number")? {
        conn.execute("ALTER TABLE appeals ADD COLUMN incoming_number TEXT;", [])?;
    }
    Ok(())
}

fn table_exists(conn: &Connection, table: &str) -> rusqlite::Result<bool> {
    let mut stmt = conn.prepare(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?1 LIMIT 1",
    )?;
    Ok(stmt.query_row([table], |_| Ok(()).map(|_| true)).optional()?.is_some())
}

fn has_column(conn: &Connection, table: &str, column: &str) -> rusqlite::Result<bool> {
    let pragma = format!("PRAGMA table_info({})", table);
    let mut stmt = conn.prepare(&pragma)?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let name: String = row.get(1)?;
        if name == column {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn open_user_db(app: &AppHandle) -> AppResult<Connection> {
    Connection::open(user_db_path(app)?).map_err(AppError::from)
}

pub fn work_dir(app: &AppHandle) -> AppResult<PathBuf> {
    let config_dir = app
        .path()
        .app_config_dir()
        .map_err(|err| AppError::new("errors.config.resolveFailed", err.to_string()))?;
    let config_path = config_dir.join("config.json");
    let content = fs::read_to_string(config_path).map_err(|_| AppError::config_missing())?;
    let config: AppConfig = serde_json::from_str(&content)?;
    let work_dir = config.work_dir.ok_or_else(AppError::config_missing)?;
    Ok(PathBuf::from(work_dir))
}

fn clients_db_path(app: &AppHandle) -> AppResult<PathBuf> {
    Ok(work_dir(app)?
        .join("Clients")
        .join(".db_clients")
        .join("clients.db"))
}

fn user_db_path(app: &AppHandle) -> AppResult<PathBuf> {
    Ok(work_dir(app)?
        .join(".db_user")
        .join("users.db"))
}
