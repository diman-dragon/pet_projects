use std::fs;

use tauri::AppHandle;

use crate::{
    error::{AppError, AppResult},
    models::payloads::{ClientDetailsPayload, SaveCasePayload},
    repository::{case_repository, client_repository, db},
    services::{case_structure, naming, path, trash},
};

pub fn load_client_details(app: &AppHandle, client_id: i64) -> AppResult<ClientDetailsPayload> {
    let conn = db::open_clients_db(app)?;
    let client = client_repository::get_by_id(&conn, client_id)?
        .ok_or_else(|| AppError::not_found("client"))?;
    let cases = case_repository::list_by_client(&conn, client_id)?;
    Ok(ClientDetailsPayload { client, cases })
}

pub fn save_case(app: &AppHandle, payload: SaveCasePayload) -> AppResult<()> {
    if payload.title.trim().is_empty() {
        return Err(AppError::validation("line.error.titleRequired", "title is required"));
    }
    match payload.id {
        Some(case_id) => update_case(app, case_id, payload),
        None => create_case(app, payload),
    }
}

pub fn delete_case(app: &AppHandle, case_id: i64) -> AppResult<()> {
    trash::soft_delete(app, "case", case_id)
}

fn create_case(app: &AppHandle, mut payload: SaveCasePayload) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    let status = naming::normalize_status(payload.status.as_deref().unwrap_or("ACTIVE"));
    
    // Determine folder name based on type
    let folder_name = match payload.case_type.as_deref() {
        Some("consulting") => {
            // 1. Handle defaults in payload first for consistency
            if payload.contract_date.is_none() || payload.contract_date.as_deref().unwrap_or("").trim().is_empty() {
                payload.contract_date = Some(chrono::Local::now().format("%F").to_string());
            }

            let date_str = payload.contract_date.as_deref()
                .map(|d| d.replace('-', ""))
                .map(|d| if d.len() >= 8 { d[2..8].to_string() } else { d }) // YYYYMMDD -> YYMMDD
                .unwrap_or_else(|| chrono::Local::now().format("%y%m%d").to_string());

            let yymm = if date_str.len() >= 4 { &date_str[0..4] } else { "2501" };
            let sequence = case_repository::count_by_type_in_month(&conn, None, path::CONSULTATIONS_DIR, yymm)?;
            
            let folder = naming::consulting_folder_name(&date_str, sequence, &status);
            format!("{}/{}", path::CONSULTATIONS_DIR, folder)
        }
        Some("contract") => {
            // 1. Handle defaults in payload first for consistency
            if payload.contract_date.is_none() || payload.contract_date.as_deref().unwrap_or("").trim().is_empty() {
                payload.contract_date = Some(chrono::Local::now().format("%F").to_string());
            }
            if payload.counterparty.is_none() || payload.counterparty.as_deref().unwrap_or("").trim().is_empty() {
                payload.counterparty = Some("BEZ_KONT".to_string());
            }

            let date_str = payload.contract_date.as_deref()
                .map(|d| d.replace('-', ""))
                .map(|d| if d.len() >= 8 { d[2..8].to_string() } else { d }) // YYYYMMDD -> YYMMDD
                .unwrap_or_else(|| chrono::Local::now().format("%y%m%d").to_string());

            naming::contract_folder_name(
                payload.counterparty.as_deref().unwrap_or("BEZ_KONT"),
                &date_str,
                payload.case_number.as_deref().unwrap_or(""),
                &status
            )
        }
        Some("other") => {
            // 1. Handle defaults in payload first for consistency
            if payload.contract_date.is_none() || payload.contract_date.as_deref().unwrap_or("").trim().is_empty() {
                payload.contract_date = Some(chrono::Local::now().format("%F").to_string());
            }

            let date_str = payload.contract_date.as_deref()
                .map(|d| d.replace('-', ""))
                .map(|d| if d.len() >= 8 { d[2..8].to_string() } else { d }) // YYYYMMDD -> YYMMDD
                .unwrap_or_else(|| chrono::Local::now().format("%y%m%d").to_string());

            let yymm = if date_str.len() >= 4 { &date_str[0..4] } else { "2501" };
            let sequence = case_repository::count_by_type_in_month(&conn, None, path::OTHER_DIR, yymm)?;
            
            let folder = naming::other_folder_name(&date_str, sequence, &status);
            format!("{}/{}", path::OTHER_DIR, folder)
        }
        _ => naming::case_folder_name(&payload.title, &status),
    };

    case_repository::insert(
        &conn,
        payload.client_id,
        payload.title.trim(),
        &status,
        payload.case_type.as_deref(),
        payload.litigation_subtype.as_deref(),
        payload.counterparty.as_deref(),
        payload.case_number.as_deref(),
        payload.metadata.as_deref(),
        &folder_name,
        payload.summary.as_deref(),
        payload.contract_date.as_deref(),
        payload.contract_type.as_deref(),
        payload.contract_amount.as_deref(),
        payload.metadata_description.as_deref(),
    )?;

    // Create case directory and special subdirectories if needed
    let case_path = path::case_path(db::work_dir(app)?, &payload.client_folder_name, &folder_name);
    
    // Ensure parent (CONSULT, CONTRACTS, etc) exists
    if let Some(parent) = case_path.parent() {
        fs::create_dir_all(parent)?;
    }
    
    fs::create_dir_all(&case_path)?;

    // Create subdirectories only for non-litigation cases
    if payload.case_type.as_deref() != Some("litigation") {
        case_structure::create_case_structure(
            case_path,
            payload.case_type.as_deref(),
            payload.counterparty.as_deref(),
            payload.litigation_subtype.as_deref(),
            payload.metadata.as_deref(),
        )?;
    }
    Ok(())
}

fn update_case(app: &AppHandle, case_id: i64, payload: SaveCasePayload) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    let current = case_repository::get_by_id(&conn, case_id)?
        .ok_or_else(|| AppError::not_found("case"))?;
    let status = naming::normalize_status(payload.status.as_deref().unwrap_or(&current.status));
    
    let mut folder_name = current.folder_name.clone();

    if current.case_type.as_deref() == Some("other") || current.case_type.as_deref() == Some("consulting") {
        // For "other" and "consulting" scenario, we support renaming the folder on status change (last part)
        let status_code = naming::case_status_code(&status);
        let mut parts = current.folder_name.split('_').map(|s| s.to_string()).collect::<Vec<_>>();
        
        if parts.len() >= 3 {
            // format: OTHER/YYMMDD_NN_STATUS or CONSULT/YYMMDD_NN_STATUS
            if parts[1] == "NOT" || parts[1] == "NOT_ENT" {
                parts[1] = "01".to_string();
            }
            let last_idx = parts.len() - 1;
            parts[last_idx] = status_code;
            folder_name = parts.join("_");
        } else {
            // fallback if format is unexpected
            let date_str = current.contract_date.as_deref()
                .map(|d| d.replace('-', ""))
                .map(|d| if d.len() >= 8 { d[2..8].to_string() } else { d })
                .unwrap_or_else(|| chrono::Local::now().format("%y%m%d").to_string());
            
            // We can't easily get the sequence here without more logic, 
            // so we try to keep it if it was there or just use 01 as fallback
            let prefix = if current.case_type.as_deref() == Some("other") { path::OTHER_DIR } else { path::CONSULTATIONS_DIR };
            folder_name = format!("{}/{}_{:02}_{}", prefix, date_str, 1, status_code);
        }
    }

    if current.case_type.as_deref() == Some("contract") {
        // For contracts, rename if date, number, counterparty or status changes
        let date_str = payload.contract_date.as_deref()
            .or(current.contract_date.as_deref())
            .map(|d| d.replace('-', ""))
            .map(|d| if d.len() >= 8 { d[2..8].to_string() } else { d })
            .unwrap_or_else(|| chrono::Local::now().format("%y%m%d").to_string());
            
        folder_name = naming::contract_folder_name(
            payload.counterparty.as_deref().unwrap_or(current.counterparty.as_deref().unwrap_or("BEZ_KONT")),
            &date_str,
            payload.case_number.as_deref().unwrap_or(current.case_number.as_deref().unwrap_or("")),
            &status
        );
    }

    // For litigation cases, rename folder when title or status changes
    if current.case_type.as_deref() == Some("litigation") {
        let title = payload.title.trim();
        let expected_folder = naming::case_folder_name(&title, &status);
        if current.folder_name != expected_folder {
            folder_name = expected_folder;
        }
    }

    if current.folder_name != folder_name {
        let old_path = path::case_path(db::work_dir(app)?, &payload.client_folder_name, &current.folder_name);
        let new_path = path::case_path(db::work_dir(app)?, &payload.client_folder_name, &folder_name);
        
        if let Some(parent) = new_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if old_path.exists() {
            fs::rename(old_path, new_path)?;
        }
    }

    let metadata = normalize_optional_string(payload.metadata.as_deref());
    let metadata_description = normalize_optional_string(payload.metadata_description.as_deref());
    
    case_repository::update(
        &conn,
        case_id,
        payload.title.trim(),
        &status,
        payload.case_type.as_deref(),
        payload.litigation_subtype.as_deref(),
        payload.counterparty.as_deref(),
        payload.case_number.as_deref(),
        metadata.as_deref(),
        &folder_name,
        payload.summary.as_deref(),
        payload.contract_date.as_deref(),
        payload.contract_type.as_deref(),
        payload.contract_amount.as_deref(),
        metadata_description.as_deref(),
    )?;
    Ok(())
}

fn normalize_optional_string(value: Option<&str>) -> Option<String> {
    value
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
}

