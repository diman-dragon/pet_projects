use std::fs;

use tauri::AppHandle;

use crate::{
    error::{AppError, AppResult},
    models::domain::TrashItem,
    repository::{db, trash_repository},
    services::path,
};

pub fn list_items(app: &AppHandle) -> AppResult<Vec<TrashItem>> {
    let conn = db::open_clients_db(app)?;
    trash_repository::list_all(&conn).map_err(AppError::from)
}

pub fn soft_delete(app: &AppHandle, entity_type: &str, entity_id: i64) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    trash_repository::soft_delete(&conn, entity_type, entity_id)?;
    Ok(())
}

pub fn restore_item(app: &AppHandle, entity_type: &str, entity_id: i64) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    if !trash_repository::is_deleted(&conn, entity_type, entity_id)? {
        return Err(AppError::validation("errors.trash.notDeleted", "entity is not deleted"));
    }
    if trash_repository::parent_deleted(&conn, entity_type, entity_id)? {
        return Err(AppError::validation("errors.trash.parentDeleted", "parent entity is deleted"));
    }
    trash_repository::restore(&conn, entity_type, entity_id)?;
    Ok(())
}

pub fn purge_item(app: &AppHandle, entity_type: &str, entity_id: i64) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    let target_path = resolve_target_path(app, &conn, entity_type, entity_id)?;
    trash_repository::purge(&conn, entity_type, entity_id)?;
    if let Some(path) = target_path {
        remove_dir_if_exists(&path)?;
    }
    Ok(())
}

fn resolve_target_path(
    app: &AppHandle,
    conn: &rusqlite::Connection,
    entity_type: &str,
    entity_id: i64,
) -> AppResult<Option<std::path::PathBuf>> {
    let work_dir = db::work_dir(app)?;
    let value = match entity_type {
        "client" => trash_repository::client_path_parts(conn, entity_id)?
            .map(|(client_folder,)| work_dir.join(path::CLIENTS_DIR).join(client_folder)),
        "case" => trash_repository::case_path_parts(conn, entity_id)?
            .map(|(client_folder, case_folder)| path::case_path(work_dir, &client_folder, &case_folder)),
        "line" => trash_repository::line_path_parts(conn, entity_id)?
            .map(|(client_folder, case_folder, line_folder, case_type)| {
                path::line_path(work_dir, &client_folder, &case_folder, &line_folder, case_type.as_deref())
            }),
        "appeal" => trash_repository::appeal_path_parts(conn, entity_id)?
            .map(|(client_folder, case_folder, line_folder, appeal_folder, case_type)| {
                path::appeal_path(work_dir, &client_folder, &case_folder, &line_folder, &appeal_folder, case_type.as_deref())
            }),
        _ => None,
    };
    Ok(value)
}

fn remove_dir_if_exists(target: &std::path::Path) -> AppResult<()> {
    if target.exists() {
        fs::remove_dir_all(target)?;
    }
    Ok(())
}
