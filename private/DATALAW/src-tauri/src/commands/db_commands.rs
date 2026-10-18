use std::{collections::HashMap, fs, path::PathBuf, sync::Mutex};

use base64::Engine;
use rusqlite::{
    params_from_iter,
    types::{Value as SqlValue, ValueRef},
    Connection,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use tauri::State;

use crate::error::{AppError, AppResult};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatabaseCommandRequest {
    pub db_path: String,
    pub sql: String,
    #[serde(default)]
    pub params: Vec<Value>,
    #[serde(default)]
    pub key_hex: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatabaseExecuteResponse {
    pub changed: usize,
    pub last_insert_id: i64,
}

#[derive(Debug, Serialize)]
pub struct DatabaseSelectResponse {
    pub rows: Vec<Value>,
}

#[derive(Default)]
pub struct DbConnectionStore {
    connections: Mutex<HashMap<String, Connection>>,
}

#[tauri::command]
pub fn db_initialize(request: DatabaseCommandRequest, state: State<'_, DbConnectionStore>) -> AppResult<()> {
    let path = PathBuf::from(&request.db_path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(AppError::from)?;
        }
    }

    let mut connections = state.connections.lock().map_err(|err| AppError::new("errors.db.lockFailed", err.to_string()))?;
    if !connections.contains_key(&request.db_path) {
        let conn = connect(&path, request.key_hex.as_deref())?;
        connections.insert(request.db_path.clone(), conn);
    }
    Ok(())
}

#[tauri::command]
pub fn db_execute(request: DatabaseCommandRequest, state: State<'_, DbConnectionStore>) -> AppResult<DatabaseExecuteResponse> {
    let mut connections = state.connections.lock().map_err(|err| AppError::new("errors.db.lockFailed", err.to_string()))?;
    ensure_connection(&mut connections, &request)?;
    let conn = connections.get_mut(&request.db_path).ok_or_else(|| AppError::new("errors.db.connectionMissing", "Database connection is not initialized"))?;

    let changed = if request.params.is_empty() && may_be_batch(&request.sql) {
        conn.execute_batch(&request.sql)?;
        0
    } else {
        let params = to_sql_params(&request.params);
        conn.execute(&request.sql, params_from_iter(params.iter()))?
    };
    let last_insert_id = conn.last_insert_rowid();
    Ok(DatabaseExecuteResponse { changed, last_insert_id })
}

#[tauri::command]
pub fn db_select(request: DatabaseCommandRequest, state: State<'_, DbConnectionStore>) -> AppResult<DatabaseSelectResponse> {
    let mut connections = state.connections.lock().map_err(|err| AppError::new("errors.db.lockFailed", err.to_string()))?;
    ensure_connection(&mut connections, &request)?;
    let conn = connections.get_mut(&request.db_path).ok_or_else(|| AppError::new("errors.db.connectionMissing", "Database connection is not initialized"))?;

    let params = to_sql_params(&request.params);
    let mut stmt = conn.prepare(&request.sql)?;
    let column_names: Vec<String> = stmt.column_names().iter().map(|name| name.to_string()).collect();
    let mut rows = stmt.query(params_from_iter(params.iter()))?;

    let mut result = Vec::new();

    while let Some(row) = rows.next()? {
        let mut object = Map::new();
        for (idx, column_name) in column_names.iter().enumerate() {
            let value = row.get_ref_unwrap(idx);
            object.insert(column_name.clone(), sql_value_to_json(value));
        }
        result.push(Value::Object(object));
    }

    Ok(DatabaseSelectResponse { rows: result })
}

#[tauri::command]
pub fn db_close(request: DatabaseCommandRequest, state: State<'_, DbConnectionStore>) -> AppResult<()> {
    let mut connections = state.connections.lock().map_err(|err| AppError::new("errors.db.lockFailed", err.to_string()))?;
    connections.remove(&request.db_path);
    Ok(())
}

fn connect(path: &PathBuf, key_hex: Option<&str>) -> AppResult<Connection> {
    let conn = Connection::open(path).map_err(AppError::from)?;
    if let Some(key) = key_hex {
        if !key.trim().is_empty() {
            if !key.chars().all(|ch| ch.is_ascii_hexdigit()) {
                return Err(AppError::validation("errors.security.invalidKey", "SQLCipher key must be hex encoded"));
            }
            let pragma = format!("PRAGMA key = \"x'{}'\";", key);
            conn.execute_batch(&pragma).map_err(AppError::from)?;
        }
    }

    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;")
        .map_err(AppError::from)?;
    Ok(conn)
}

fn ensure_connection(connections: &mut HashMap<String, Connection>, request: &DatabaseCommandRequest) -> AppResult<()> {
    if connections.contains_key(&request.db_path) {
        return Ok(());
    }

    let conn = connect(&PathBuf::from(&request.db_path), request.key_hex.as_deref())?;
    connections.insert(request.db_path.clone(), conn);
    Ok(())
}

fn may_be_batch(sql: &str) -> bool {
    let trimmed = sql.trim();
    trimmed.contains(';') || trimmed.lines().count() > 1
}

fn to_sql_params(values: &[Value]) -> Vec<SqlValue> {
    values.iter().map(sql_value_to_rusqlite).collect()
}

fn sql_value_to_rusqlite(value: &Value) -> SqlValue {
    match value {
        Value::Null => SqlValue::Null,
        Value::Bool(flag) => SqlValue::Integer(if *flag { 1 } else { 0 }),
        Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                SqlValue::Integer(integer)
            } else if let Some(unsigned) = number.as_u64() {
                SqlValue::Integer(unsigned as i64)
            } else if let Some(float) = number.as_f64() {
                SqlValue::Real(float)
            } else {
                SqlValue::Null
            }
        }
        Value::String(text) => SqlValue::Text(text.clone()),
        Value::Array(items) => SqlValue::Text(serde_json::to_string(items).unwrap_or_default()),
        Value::Object(map) => SqlValue::Text(serde_json::to_string(map).unwrap_or_default()),
    }
}

fn sql_value_to_json(value: ValueRef<'_>) -> Value {
    match value {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(integer) => Value::Number(Number::from(integer)),
        ValueRef::Real(real) => Value::Number(Number::from_f64(real).unwrap_or_else(|| Number::from(0))),
        ValueRef::Text(text) => Value::String(String::from_utf8_lossy(text).into_owned()),
        ValueRef::Blob(bytes) => Value::String(base64::engine::general_purpose::STANDARD.encode(bytes)),
    }
}
