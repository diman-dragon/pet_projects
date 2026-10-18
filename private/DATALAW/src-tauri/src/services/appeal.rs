use tauri::AppHandle;

use crate::{
    error::{AppError, AppResult},
    models::payloads::AppealDetailsPayload,
    repository::{appeal_repository, db},
    services::path,
};

pub fn load_appeal_details(app: &AppHandle, appeal_id: i64) -> AppResult<AppealDetailsPayload> {
    let conn = db::open_clients_db(app)?;
    let appeal = appeal_repository::get_by_id(&conn, appeal_id)?
        .ok_or_else(|| AppError::not_found("appeal"))?;
    let summary = appeal_repository::get_summary(&conn, appeal_id)?;
    let documents = appeal_repository::get_documents(&conn, appeal_id)?;
    let chain_context = appeal_repository::get_chain_context(&conn, appeal_id)?;
    let folder_path = build_folder_path(app, &conn, appeal_id, &appeal.folder_name)?;
    Ok(AppealDetailsPayload {
        appeal,
        summary,
        documents,
        folder_path,
        chain_context,
    })
}

fn build_folder_path(
    app: &AppHandle,
    conn: &rusqlite::Connection,
    appeal_id: i64,
    folder_name: &str,
) -> AppResult<Option<String>> {
    let Some((client_folder, case_folder, line_folder, case_type)) = appeal_repository::get_path_data(conn, appeal_id)? else {
        return Ok(None);
    };
    let path = path::appeal_path(db::work_dir(app)?, &client_folder, &case_folder, &line_folder, folder_name, case_type.as_deref());
    Ok(Some(path.to_string_lossy().to_string()))
}
