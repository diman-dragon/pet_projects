use std::fs;

use chrono::Local;
use tauri::AppHandle;

use crate::{
    error::{AppError, AppResult},
    models::payloads::{CaseDetailsPayload, LineDetailsPayload, SaveAppealPayload, SaveLinePayload},
    repository::{appeal_repository, case_repository, client_repository, db, line_repository},
    services::{agency_codes, naming, path, trash},
};

pub fn load_case_details(app: &AppHandle, case_id: i64) -> AppResult<CaseDetailsPayload> {
    let conn = db::open_clients_db(app)?;
    let case_item = case_repository::get_by_id(&conn, case_id)?
        .ok_or_else(|| AppError::not_found("case"))?;
    let lines = line_repository::list_by_case(&conn, case_id)?;
    Ok(CaseDetailsPayload { case_item, lines })
}

pub fn save_line(app: &AppHandle, payload: SaveLinePayload) -> AppResult<()> {
    if payload.title.trim().is_empty() {
        return Err(AppError::validation("line.error.titleRequired", "title is required"));
    }
    match payload.id {
        Some(line_id) => update_line(app, line_id, payload),
        None => create_line(app, payload),
    }
}

pub fn delete_line(app: &AppHandle, line_id: i64) -> AppResult<()> {
    trash::soft_delete(app, "line", line_id)
}

pub fn load_line_details(app: &AppHandle, line_id: i64) -> AppResult<LineDetailsPayload> {
    let conn = db::open_clients_db(app)?;
    let line = line_repository::get_by_id(&conn, line_id)?
        .ok_or_else(|| AppError::not_found("line"))?;
    let appeals = appeal_repository::list_by_line(&conn, line_id)?;
    Ok(LineDetailsPayload { line, appeals })
}

pub fn save_appeal(app: &AppHandle, payload: SaveAppealPayload) -> AppResult<i64> {
    if payload.title.trim().is_empty() {
        return Err(AppError::validation("line.error.titleRequired", "title is required"));
    }
    match payload.id {
        Some(appeal_id) => {
            update_appeal(app, appeal_id, payload.clone())?;
            Ok(appeal_id)
        }
        None => create_appeal(app, payload),
    }
}

pub fn delete_appeal(app: &AppHandle, appeal_id: i64) -> AppResult<()> {
    trash::soft_delete(app, "appeal", appeal_id)
}

fn create_line(app: &AppHandle, payload: SaveLinePayload) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    let case_item = case_repository::get_by_id(&conn, payload.case_id)?
        .ok_or_else(|| AppError::not_found("case"))?;
    
    // FIX: Check if line with same title already exists for this case (prevent duplicates)
    if let Some(_existing) = line_repository::get_by_case_and_title(&conn, payload.case_id, &payload.title)? {
        log::info!(
            "Line with title '{}' already exists for case {}, skipping creation",
            payload.title,
            payload.case_id
        );
        return Ok(());
    }
    
    let client_folder = resolve_client_folder(&conn, case_item.client_id, &payload.client_folder_name)?;
    let folder_name = naming::to_standard_name(&payload.title);
    line_repository::insert(
        &conn,
        payload.case_id,
        payload.title.trim(),
        payload.status.as_deref(),
        &folder_name,
        payload.description.as_deref(),
        payload.consultation_text.as_deref(),
        payload.consultation_topic.as_deref(),
        payload.contract_analysis.as_deref(),
        payload.contract_risks.as_deref(),
    )?;

    // Skip folder creation for litigation
    if payload.case_type.as_deref() == Some("litigation") {
        return Ok(());
    }

    let base_path = path::line_path(
        db::work_dir(app)?,
        &client_folder,
        &payload.case_folder_name,
        &folder_name,
        payload.case_type.as_deref(),
    );
    fs::create_dir_all(&base_path)?;

    // Save to file if needed
    if let Some(case_type) = payload.case_type.as_deref() {
        match case_type {
            "consulting" => {
                if let Some(text) = payload.consultation_text {
                    let file_path = base_path.join("CONSULTATION.txt");
                    fs::write(file_path, text)?;
                }
            }
            "other" => {
                if let Some(text) = payload.description {
                    let file_path = base_path.join("NOTES.txt");
                    fs::write(file_path, text)?;
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn update_line(app: &AppHandle, line_id: i64, payload: SaveLinePayload) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    line_repository::update(
        &conn,
        line_id,
        payload.title.trim(),
        payload.status.as_deref(),
        payload.description.as_deref(),
        payload.consultation_text.as_deref(),
        payload.consultation_topic.as_deref(),
        payload.contract_analysis.as_deref(),
        payload.contract_risks.as_deref(),
    )?;

    // Save to file if needed
    if let Some(case_type) = payload.case_type.as_deref() {
        let base_path = path::line_path(
            db::work_dir(app)?,
            &payload.client_folder_name,
            &payload.case_folder_name,
            "", // line_folder is not used for flat types
            Some(case_type),
        );

        match case_type {
            "consulting" => {
                if let Some(text) = payload.consultation_text {
                    let file_path = base_path.join("CONSULTATION.txt");
                    fs::write(file_path, text)?;
                }
            }
            "other" => {
                if let Some(text) = payload.description {
                    let file_path = base_path.join("NOTES.txt");
                    fs::write(file_path, text)?;
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn create_appeal(app: &AppHandle, payload: SaveAppealPayload) -> AppResult<i64> {
    let conn = db::open_clients_db(app)?;
    let status = naming::normalize_status(payload.status.as_deref().unwrap_or("DRAFT"));
    let sequence = appeal_repository::list_by_line(&conn, payload.line_id)?.len() as i64 + 1;
    let resolved_agency_code = match payload.agency_code.as_deref().map(str::trim) {
        Some(value) if !value.is_empty() => Some(agency_codes::resolve_agency_code(app, value)?),
        _ => None,
    };
    // For litigation: always use today's date, otherwise use provided date or today
    let appeal_date = if payload.case_type.as_deref() == Some("litigation") {
        Local::now().format("%F").to_string()
    } else {
        payload.appeal_date.unwrap_or_else(|| Local::now().format("%F").to_string())
    };
    let folder_name = naming::appeal_folder_name(
        &payload.title,
        resolved_agency_code.as_deref(),
        &status,
        sequence,
        payload.case_type.as_deref(),
        Some(&appeal_date),
        payload.incoming_number.as_deref(),
    );
    let id = appeal_repository::insert(
        &conn,
        payload.line_id,
        payload.title.trim(),
        &status,
        resolved_agency_code.as_deref(),
        Some(&appeal_date),
        payload.incoming_number.as_deref(),
        payload.outgoing_number.as_deref(),
        &folder_name,
    )?;
    fs::create_dir_all(path::appeal_path(
        db::work_dir(app)?,
        &payload.client_folder_name,
        &payload.case_folder_name,
        &payload.line_folder_name,
        &folder_name,
        payload.case_type.as_deref(),
    ))?;
    Ok(id)
}

fn resolve_client_folder(conn: &rusqlite::Connection, client_id: i64, fallback: &str) -> AppResult<String> {
    if !fallback.is_empty() {
        return Ok(fallback.to_string());
    }
    client_repository::get_folder_name(conn, client_id)?
        .ok_or_else(|| AppError::not_found("client"))
}

fn update_appeal(app: &AppHandle, appeal_id: i64, payload: SaveAppealPayload) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    let current = appeal_repository::get_by_id(&conn, appeal_id)?
        .ok_or_else(|| AppError::not_found("appeal"))?;
    let resolved_agency_code = match payload.agency_code.as_deref().map(str::trim) {
        Some(value) if !value.is_empty() => Some(agency_codes::resolve_agency_code(app, value)?),
        _ => None,
    };

    // For litigation: prevent changing agency_code and title after first save
    if payload.case_type.as_deref() == Some("litigation") {
        if let Some(new_agency) = resolved_agency_code.as_deref() {
            if let Some(current_agency) = current.agency_code.as_deref() {
                if !current_agency.is_empty() && new_agency != current_agency {
                    return Err(AppError::validation("appeal.error.agencyImmutable", "Agency code cannot be changed after first save"));
                }
            }
        }
        if payload.title.trim() != current.title {
            return Err(AppError::validation("appeal.error.titleImmutable", "Title cannot be changed after first save"));
        }
    }

    let status = naming::normalize_status(payload.status.as_deref().unwrap_or("DRAFT"));
    let appeal_date = payload.appeal_date.clone().unwrap_or_else(|| Local::now().format("%F").to_string());
    let folder_name = if payload.case_type.as_deref() == Some("other") {
        naming::appeal_folder_name(
            &payload.title,
            resolved_agency_code.as_deref(),
            &status,
            0,
            payload.case_type.as_deref(),
            Some(&appeal_date),
            payload.incoming_number.as_deref(),
        )
    } else if payload.case_type.as_deref() == Some("litigation") {
        // For litigation: always regenerate folder name with current agency_code and status
        naming::appeal_folder_name(
            &payload.title,
            resolved_agency_code.as_deref(),
            &status,
            0,
            payload.case_type.as_deref(),
            Some(&appeal_date),
            payload.incoming_number.as_deref(),
        )
    } else {
        let sequence = current
            .folder_name
            .split('_')
            .nth(1)
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(1);
        naming::appeal_folder_name(
            &payload.title,
            resolved_agency_code.as_deref(),
            &status,
            sequence,
            payload.case_type.as_deref(),
            Some(&appeal_date),
            payload.incoming_number.as_deref(),
        )
    };
    if current.folder_name != folder_name && !["consulting", "contract"].contains(&payload.case_type.as_deref().unwrap_or("")) {
        let old_path = path::appeal_path(
            db::work_dir(app)?,
            &payload.client_folder_name,
            &payload.case_folder_name,
            &payload.line_folder_name,
            &current.folder_name,
            payload.case_type.as_deref(),
        );
        let new_path = path::appeal_path(
            db::work_dir(app)?,
            &payload.client_folder_name,
            &payload.case_folder_name,
            &payload.line_folder_name,
            &folder_name,
            payload.case_type.as_deref(),
        );
        if let Some(parent) = new_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if old_path.exists() {
            fs::rename(old_path, new_path)?;
        }
    }
    appeal_repository::update(
        &conn,
        appeal_id,
        payload.title.trim(),
        &status,
        resolved_agency_code.as_deref(),
        payload.appeal_date.as_deref(),
        payload.incoming_number.as_deref(),
        payload.outgoing_number.as_deref(),
        &folder_name,
    )?;
    Ok(())
}
