use std::fs;

use tauri::AppHandle;

use crate::{
    error::{AppError, AppResult},
    models::{domain::Client, payloads::SaveClientPayload},
    repository::{client_repository, db},
    services::{naming, path, trash},
};

pub fn list_clients(app: &AppHandle) -> AppResult<Vec<Client>> {
    let conn = db::open_clients_db(app)?;
    client_repository::list_all(&conn).map_err(AppError::from)
}

pub fn save_client(app: &AppHandle, payload: SaveClientPayload) -> AppResult<()> {
    if payload.full_name.trim().is_empty() {
        return Err(AppError::validation("client.error.nameRequired", "full_name is required"));
    }

    match payload.id {
        Some(client_id) => update_client(app, client_id, payload),
        None => create_client(app, payload),
    }
}

pub fn delete_client(app: &AppHandle, client_id: i64) -> AppResult<()> {
    trash::soft_delete(app, "client", client_id)
}

fn create_client(app: &AppHandle, payload: SaveClientPayload) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    let client_type = normalize_client_type(&payload.client_type);
    let status = naming::normalize_status(&payload.status);
    let sequence = client_repository::count_all(&conn)?;
    let folder_name = naming::client_folder_name(payload.full_name.trim(), &client_type, &status, sequence);
    let tax_id = normalize_optional_string(payload.tax_id.as_deref());
    let email = normalize_optional_string(payload.email.as_deref());
    let metadata = normalize_optional_string(payload.metadata.as_deref());
    client_repository::insert(
        &conn,
        payload.full_name.trim(),
        tax_id.as_deref(),
        email.as_deref(),
        &client_type,
        &status,
        &folder_name,
        metadata.as_deref(),
    )?;
    fs::create_dir_all(db::work_dir(app)?.join(path::CLIENTS_DIR).join(&folder_name).join(path::CLIENT_BASE_DIR))?;
    Ok(())
}

fn update_client(app: &AppHandle, client_id: i64, payload: SaveClientPayload) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    let current = client_repository::get_by_id(&conn, client_id)?.ok_or_else(|| AppError::not_found("client"))?;
    let client_type = normalize_client_type(&payload.client_type);
    let status = naming::normalize_status(&payload.status);
    let folder_name = naming::rename_client_folder(&current.folder_name, &status);
    let tax_id = normalize_optional_string(payload.tax_id.as_deref());
    let email = normalize_optional_string(payload.email.as_deref());
    let metadata = normalize_optional_string(payload.metadata.as_deref());
    rename_client_folder(app, &current.folder_name, &folder_name)?;
    client_repository::update(
        &conn,
        client_id,
        payload.full_name.trim(),
        tax_id.as_deref(),
        email.as_deref(),
        &client_type,
        &status,
        &folder_name,
        metadata.as_deref(),
    )?;
    Ok(())
}

fn rename_client_folder(app: &AppHandle, previous: &str, next: &str) -> AppResult<()> {
    if previous == next {
        return Ok(());
    }
    fs::rename(
        db::work_dir(app)?.join(path::CLIENTS_DIR).join(previous),
        db::work_dir(app)?.join(path::CLIENTS_DIR).join(next),
    )?;
    Ok(())
}

fn normalize_client_type(value: &str) -> String {
    match value.to_uppercase().as_str() {
        "INDIVIDUAL" | "IP" => "IP".to_string(),
        "LEGAL" | "LE" => "LE".to_string(),
        "NGO" | "NO" => "NO".to_string(),
        other => other.to_string(),
    }
}

fn normalize_optional_string(value: Option<&str>) -> Option<String> {
    value
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
}
