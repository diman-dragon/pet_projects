use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use chrono::{Local, NaiveDate};
use regex::Regex;
use serde_json::json;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter};
use zip::ZipArchive;

use crate::{
    error::{AppError, AppResult},
    models::payloads::{AppealWorkbenchPayload, DeadlineEventPayload, ScannedFilePayload, SummaryGenerationOptions, UploadDocumentPayload},
    repository::{appeal_repository, case_repository, db, line_repository},
    services::{ai, appeal, path, naming},
};

pub fn load_appeal_workbench(app: &AppHandle, appeal_id: i64) -> AppResult<AppealWorkbenchPayload> {
    let details = appeal::load_appeal_details(app, appeal_id)?;
    let folder_path = details
        .folder_path
        .clone()
        .ok_or_else(|| AppError::new("errors.fs.pathMissing", "appeal folder path missing"))?;
    let files = scan_folder(&folder_path)?;
    let deadlines = build_deadlines(details.appeal.appeal_date.as_deref(), &files);
    let tracked = details
        .documents
        .iter()
        .map(|doc| doc.file_path.clone())
        .collect::<std::collections::HashSet<_>>();
    let pending_retro = files.iter().any(|file| !tracked.contains(&file.path));
    Ok(AppealWorkbenchPayload {
        details,
        files,
        deadlines,
        pending_retro,
        last_scanned_at: Local::now().to_rfc3339(),
    })
}

pub fn rename_appeal_file(app: &AppHandle, file_path: &str, new_stem: &str) -> AppResult<()> {
    let source = PathBuf::from(file_path);
    let extension = source.extension().and_then(|value| value.to_str()).unwrap_or_default();
    let target = source.with_file_name(if extension.is_empty() {
        new_stem.to_string()
    } else {
        format!("{new_stem}.{extension}")
    });
    fs::rename(source, target)?;
    let _ = app;
    Ok(())
}

pub fn upload_appeal_document(app: &AppHandle, appeal_id: i64, source_path: &str) -> AppResult<AppealWorkbenchPayload> {
    let details = appeal::load_appeal_details(app, appeal_id)?;
    let folder = details
        .folder_path
        .clone()
        .ok_or_else(|| AppError::new("errors.fs.pathMissing", "appeal folder path missing"))?;
    let source = PathBuf::from(source_path);
    let file_name = source
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| AppError::new("errors.fs.invalidFileName", "invalid file name"))?
        .to_string();
    let target = PathBuf::from(&folder).join(&file_name);
    if source != target {
        fs::copy(&source, &target)?;
    }
    process_document(app, appeal_id, &details, &target)?;
    load_appeal_workbench(app, appeal_id)
}

pub fn upload_document(
    app: &AppHandle,
    payload: UploadDocumentPayload,
) -> AppResult<String> {
    let base_path = path::appeal_path(
        db::work_dir(app)?,
        &payload.client_folder_name,
        &payload.case_folder_name,
        &payload.line_folder_name,
        &payload.appeal_folder_name,
        payload.case_type.as_deref(),
    );

    // Create directory if it doesn't exist (for non-flat types)
    fs::create_dir_all(&base_path)?;

    let ext = Path::new(&payload.file_name)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let stem = Path::new(&payload.file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document");
    
    let standard_name = if ext.is_empty() {
        naming::to_standard_name(stem)
    } else {
        format!("{}.{}", naming::to_standard_name(stem), ext)
    };

    let destination = base_path.join(&standard_name);

    fs::copy(&payload.file_path, &destination)?;

    Ok(standard_name)
}

pub fn retro_process_appeal_documents(app: &AppHandle, appeal_id: i64) -> AppResult<AppealWorkbenchPayload> {
    let details = appeal::load_appeal_details(app, appeal_id)?;
    let folder = details
        .folder_path
        .clone()
        .ok_or_else(|| AppError::new("errors.fs.pathMissing", "appeal folder path missing"))?;
    let existing = details
        .documents
        .iter()
        .map(|doc| doc.file_path.clone())
        .collect::<std::collections::HashSet<_>>();
    for entry in fs::read_dir(folder)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() || is_summary_file(&path) {
            continue;
        }
        let path_string = path.to_string_lossy().to_string();
        if existing.contains(&path_string) {
            continue;
        }
        process_document(app, appeal_id, &details, &path)?;
    }
    load_appeal_workbench(app, appeal_id)
}

fn is_summary_file(path: &Path) -> bool {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    name.contains(".ai_summary.txt") || name == "AI_SUMMARY.txt"
}

pub fn reanalyze_document(app: &AppHandle, appeal_id: i64, file_path: &str) -> AppResult<AppealWorkbenchPayload> {
    let details = appeal::load_appeal_details(app, appeal_id)?;
    let path = PathBuf::from(file_path);
    if !path.exists() {
        return Err(AppError::new("errors.fs.fileNotFound", "file not found"));
    }
    process_document(app, appeal_id, &details, &path)?;
    load_appeal_workbench(app, appeal_id)
}

pub fn generate_case_summary(
    app: &AppHandle,
    case_id: i64,
    options: Option<SummaryGenerationOptions>,
) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    let case_item = case_repository::get_by_id(&conn, case_id)?
        .ok_or_else(|| AppError::not_found("case"))?;
    let parts = case_repository::get_line_summaries(&conn, case_id)?
        .into_iter()
        .filter_map(|(_, summary)| summary)
        .collect::<Vec<_>>();

    let app_clone = app.clone();
    let title = case_item.title.clone();
    let case_folder = case_item.folder_name.clone();
    let client_id = case_item.client_id;

    // Get client folder name
    let client = crate::repository::client_repository::get_by_id(&conn, client_id)?
        .ok_or_else(|| AppError::not_found("client"))?;
    let client_folder = client.folder_name.clone();
    let summary_options = options.clone();

    tauri::async_runtime::spawn(async move {
        let error_app = app_clone.clone();
        let res: AppResult<()> = async move {
            let summary = if parts.is_empty() {
                "Нет данных для анализа".to_string()
            } else {
                ai::summarize_chain(
                    app_clone.clone(),
                    "case".to_string(),
                    title.clone(),
                    parts,
                    summary_options,
                )
                .await?
            };
            let conn = db::open_clients_db(&app_clone)?;
            case_repository::save_summary(&conn, case_id, &summary)?;

            // Write case summary to file
            let summary_path = std::path::PathBuf::from(&client_folder)
                .join(&case_folder)
                .join("AI_SUMMARY.txt");
            let _ = std::fs::write(&summary_path, format!("=== AI Сводка дела: {} ===\n\n{}", title, summary));

            let _ = app_clone.emit("summary-completed", json!({ "type": "case", "id": case_id, "summary": summary }));

            Ok(())
        }.await;

        if let Err(err) = res {
            eprintln!("[document] Case summary generation failed: {err:?}");
            let _ = error_app.emit("summary-error", json!({ "type": "case", "id": case_id, "error": err.to_string() }));
        }
    });

    Ok(())
}

pub fn generate_line_summary(
    app: &AppHandle,
    line_id: i64,
    options: Option<SummaryGenerationOptions>,
) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    let line = line_repository::get_by_id(&conn, line_id)?
        .ok_or_else(|| AppError::not_found("line"))?;
    let parts = line_repository::get_appeal_summaries(&conn, line_id)?
        .into_iter()
        .filter_map(|(_, summary)| summary)
        .collect::<Vec<_>>();

    let app_clone = app.clone();
    let title = line.title.clone();
    let line_folder = line.folder_name.clone();
    let case_folder = line.case_folder.clone();
    let client_folder = line.client_folder.clone();
    let summary_options = options.clone();

    tauri::async_runtime::spawn(async move {
        let error_app = app_clone.clone();
        let res: AppResult<()> = async move {
            let summary = if parts.is_empty() {
                "Нет данных для анализа".to_string()
            } else {
                ai::summarize_chain(
                    app_clone.clone(),
                    "line".to_string(),
                    title.clone(),
                    parts,
                    summary_options,
                )
                .await?
            };
            let conn = db::open_clients_db(&app_clone)?;
            line_repository::save_summary(&conn, line_id, &summary)?;

            // Write line summary to file
            let summary_path = std::path::PathBuf::from(&client_folder)
                .join(&case_folder)
                .join(&line_folder)
                .join("AI_SUMMARY.txt");
            let _ = std::fs::write(&summary_path, format!("=== AI Сводка линии: {} ===\n\n{}", title, summary));

            let _ = app_clone.emit("summary-completed", json!({ "type": "line", "id": line_id, "summary": summary }));

            Ok(())
        }.await;

        if let Err(err) = res {
            eprintln!("[document] Line summary generation failed: {err:?}");
            let _ = error_app.emit("summary-error", json!({ "type": "line", "id": line_id, "error": err.to_string() }));
        }
    });

    Ok(())
}

pub fn generate_client_summary(
    app: &AppHandle,
    client_id: i64,
    options: Option<SummaryGenerationOptions>,
) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    let client = crate::repository::client_repository::get_by_id(&conn, client_id)?
        .ok_or_else(|| AppError::not_found("client"))?;

    let mut parts = Vec::new();

    // 1. Client Dossier (FNS/KAD) - This is the most important "base" info
    if let Some(client_data) = client.client_data.clone().filter(|value| !value.trim().is_empty()) {
        parts.push(format!("### ОФИЦИАЛЬНЫЕ ДАННЫЕ (ЕГРЮЛ/КАД):\n{client_data}"));
    }

    // 2. Case summaries (which include Line summaries, which include Appeal summaries)
    let cases = crate::repository::client_repository::get_cases_for_summary(&conn, client_id)?;
    if !cases.is_empty() {
        parts.push("### ТЕКУЩИЕ ДЕЛА И ПРОИЗВОДСТВА:".to_string());
        for (title, status, summary) in cases {
            let status_label = match status.as_str() {
                "ACTIVE" => "В работе",
                "ARCHIVE" => "Архив",
                "LEAD" => "Лид",
                _ => &status
            };
            parts.push(format!("- [{status_label}] {title}: {}", summary.unwrap_or_else(|| "Нет сводки по делу".to_string())));
        }
    }

    let app_clone = app.clone();
    let full_name = client.full_name.clone();
    let client_folder = client.folder_name.clone();
    let summary_options = options.clone();

    tauri::async_runtime::spawn(async move {
        let error_app = app_clone.clone();
        let res: AppResult<()> = async move {
            let summary = if parts.is_empty() {
                "Нет данных для анализа клиента".to_string()
            } else {
                ai::summarize_chain(
                    app_clone.clone(),
                    "client".to_string(),
                    full_name.clone(),
                    parts,
                    summary_options,
                )
                .await?
            };
            let conn = db::open_clients_db(&app_clone)?;
            crate::repository::client_repository::save_summary(&conn, client_id, &summary)?;

            // Write client summary to file
            let summary_path = std::path::PathBuf::from(&client_folder).join("AI_SUMMARY.txt");
            let _ = std::fs::write(&summary_path, format!("=== AI Сводка клиента: {} ===\n\n{}", full_name, summary));

            let _ = app_clone.emit("summary-completed", json!({ "type": "client", "id": client_id, "summary": summary }));

            Ok(())
        }.await;

        if let Err(err) = res {
            eprintln!("[document] Client summary generation failed: {err:?}");
            let _ = error_app.emit("summary-error", json!({ "type": "client", "id": client_id, "error": err.to_string() }));
        }
    });

    Ok(())
}

fn process_document(
    app: &AppHandle,
    appeal_id: i64,
    details: &crate::models::payloads::AppealDetailsPayload,
    target_path: &Path,
) -> AppResult<()> {
    let bytes = fs::read(target_path)?;
    let file_hash = compute_hash(&bytes);
    let conn = db::open_clients_db(app)?;

    // Skip if this exact file path is already registered for the appeal.
    let path_str = target_path.to_string_lossy();
    if appeal_repository::exists_by_path(&conn, appeal_id, &path_str)? {
        return Ok(());
    }

    let duplicate = appeal_repository::find_duplicate(&conn, &file_hash, appeal_id)?;
    let file_name = target_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("document")
        .to_string();

    // Insert immediately with placeholder summary to provide instant feedback.
    appeal_repository::insert_document(
        &conn,
        appeal_id,
        &file_name,
        &target_path.to_string_lossy(),
        &file_hash,
        Some("Анализ документа..."),
        None,
        duplicate.as_ref().map(|item| item.0.as_str()),
        duplicate.as_ref().map(|item| item.1),
    )?;

    // Spawn processing in background using tauri async runtime
    let app_clone = app.clone();
    let details_clone = details.clone();
    let path_buf = target_path.to_path_buf();
    let file_name_clone = file_name.clone();
    let target_path_clone = target_path.to_path_buf();

    tauri::async_runtime::spawn(async move {
        let res: AppResult<()> = async move {
            let extracted = extract_text(&path_buf).unwrap_or_else(|_| String::new());
            let (summary, key_points) = if extracted.trim().len() > 20 {
                ai::summarize_document(app_clone.clone(), file_name_clone.clone(), extracted, None).await?
            } else {
                ("Нет текста для анализа".to_string(), Vec::new())
            };

            let key_points_json = if key_points.is_empty() {
                None
            } else {
                Some(json!(key_points).to_string())
            };

            let conn = db::open_clients_db(&app_clone)?;
            appeal_repository::update_document_analysis(
                &conn,
                appeal_id,
                &path_buf.to_string_lossy(),
                &summary,
                key_points_json.as_deref(),
            )?;
            // eprintln!("[document] Summary saved to database: path={}, summary_len={}", path_buf.display(), summary.len());

            // Write document summary to file
            let summary_path = target_path_clone.with_extension("ai_summary.txt");
            let summary_content = format!("=== AI Сводка документа: {} ===\n\n{}", file_name_clone, summary);
            if !key_points.is_empty() {
                let key_points_str = key_points.join("\n- ");
                let _ = std::fs::write(&summary_path, format!("{}\n\nКлючевые моменты:\n- {}", summary_content, key_points_str));
            } else {
                let _ = std::fs::write(&summary_path, summary_content);
            }

            update_chain_summaries(&app_clone, &details_clone)?;

            // Notify frontend that analysis is complete
            // eprintln!("[document] Emitting analysis-completed event for appeal_id={}", appeal_id);
            let _ = app_clone.emit("analysis-completed", json!({ "appeal_id": appeal_id }));
            Ok(())
        }.await;

        if let Err(err) = res {
            eprintln!("[document] Background processing failed: {err:?}");
        }
    });

    Ok(())
}

fn update_chain_summaries(app: &AppHandle, details: &crate::models::payloads::AppealDetailsPayload) -> AppResult<()> {
    let conn = db::open_clients_db(app)?;
    let appeal_id = details.appeal.id;
    let chain = details
        .chain_context
        .clone()
        .ok_or_else(|| AppError::new("errors.appeal.chainMissing", "chain context missing"))?;
    
    // Filter out empty and placeholder summaries
    let docs = appeal_repository::get_documents(&conn, appeal_id)?;
    let doc_summaries: Vec<String> = docs
        .into_iter()
        .filter_map(|item| item.summary)
        .filter(|s| !s.trim().is_empty() && s != "Анализ документа..." && s != "Нет текста для анализа")
        .collect();
    
    eprintln!("[update_chain_summaries] appeal_id={}, doc_summaries_count={}", appeal_id, doc_summaries.len());

    let app_clone = app.clone();
    let appeal_title = details.appeal.title.clone();
    let appeal_folder = details.appeal.folder_name.clone();
    let line_title = chain.line_title.clone();
    let case_title = chain.case_title.clone();
    let line_id = chain.line_id;
    let case_id = chain.case_id;

    // Get path data for file creation
    let path_data = appeal_repository::get_path_data(&conn, appeal_id)?;
    let (client_folder, case_folder, line_folder) = if let Some((cf, caf, lf, _)) = path_data {
        (cf, caf, lf)
    } else {
        eprintln!("[update_chain_summaries] path_data is None, cannot create summary files");
        // Still update DB summaries even if we can't write files
        let client_folder = String::new();
        let case_folder = String::new();
        let line_folder = String::new();
        (client_folder, case_folder, line_folder)
    };

    tauri::async_runtime::spawn(async move {
        let res: AppResult<()> = async move {
            // Appeal summary
            let appeal_summary = ai::summarize_chain(
                app_clone.clone(),
                "appeal".to_string(),
                appeal_title.clone(),
                doc_summaries,
                None,
            )
            .await?;
            let conn = db::open_clients_db(&app_clone)?;
            appeal_repository::update_appeal_summary(&conn, appeal_id, &appeal_summary)?;

            // Write appeal summary to file
            let summary_path = std::path::PathBuf::from(&client_folder)
                .join(&case_folder)
                .join(&line_folder)
                .join(&appeal_folder)
                .join("AI_SUMMARY.txt");
            let _ = std::fs::write(&summary_path, &appeal_summary);

            // Line summary
            let line_parts = line_repository::get_appeal_summaries(&conn, line_id)?
                .into_iter()
                .filter_map(|(_, summary)| summary)
                .filter(|s| !s.trim().is_empty() && s != "Анализ документа..." && s != "Нет текста для анализа" && s != "Нет данных для анализа")
                .collect::<Vec<_>>();
            eprintln!("[update_chain_summaries] line_id={}, line_parts_count={}", line_id, line_parts.len());
            let line_summary = ai::summarize_chain(
                app_clone.clone(),
                "line".to_string(),
                line_title.clone(),
                line_parts,
                None,
            )
            .await?;
            line_repository::save_summary(&conn, line_id, &line_summary)?;

            // Write line summary to file
            let line_summary_path = std::path::PathBuf::from(&client_folder)
                .join(&case_folder)
                .join(&line_folder)
                .join("AI_SUMMARY.txt");
            let _ = std::fs::write(&line_summary_path, &line_summary);

            // Case summary
            let case_parts = case_repository::get_line_summaries(&conn, case_id)?
                .into_iter()
                .filter_map(|(_, summary)| summary)
                .collect::<Vec<_>>();
            let case_summary = ai::summarize_chain(
                app_clone.clone(),
                "case".to_string(),
                case_title.clone(),
                case_parts,
                None,
            )
            .await?;
            case_repository::save_summary(&conn, case_id, &case_summary)?;

            // Write case summary to file
            let case_summary_path = std::path::PathBuf::from(&client_folder)
                .join(&case_folder)
                .join("AI_SUMMARY.txt");
            let _ = std::fs::write(&case_summary_path, &case_summary);

            // Notify frontend that chain summary update is complete
            eprintln!("[document] Emitting chain-summary-completed event for appeal_id={}", appeal_id);
            let _ = app_clone.emit("chain-summary-completed", json!({ "appeal_id": appeal_id }));

            Ok(())
        }.await;

        if let Err(err) = res {
            eprintln!("[document] Chain summary update failed: {err:?}");
        }
    });

    Ok(())
}

fn scan_folder(folder_path: &str) -> AppResult<Vec<ScannedFilePayload>> {
    let mut files = fs::read_dir(folder_path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| !is_summary_file(&entry.path()))
        .filter_map(|entry| to_scanned_file(entry.path()).ok())
        .collect::<Vec<_>>();
    files.sort_by(|left, right| right.parsed_date.cmp(&left.parsed_date).then(left.name.cmp(&right.name)));
    Ok(files)
}

fn to_scanned_file(path: PathBuf) -> AppResult<ScannedFilePayload> {
    let metadata = fs::metadata(&path)?;
    let name = path.file_name().and_then(|value| value.to_str()).unwrap_or_default().to_string();
    let ext = path.extension().and_then(|value| value.to_str()).map(|value| format!(".{}", value.to_lowercase())).unwrap_or_default();
    let stem = path.file_stem().and_then(|value| value.to_str()).unwrap_or_default().to_string();
    let pattern = Regex::new(r"^(\d{6})_([A-Za-z][A-Za-z0-9_]*)_(v\d{2,})_(ACT|CLS|PEND|DRAFT)$")
        .map_err(|err| AppError::new("errors.regex.invalid", err.to_string()))?;
    let captures = pattern.captures(&stem);
    let is_standard = captures.is_some();
    let (date_stamp, parsed_date, description, version, status) = if let Some(ref caps) = captures {
        let date_stamp = caps.get(1).map(|value| value.as_str().to_string());
        let parsed_date = date_stamp
            .as_deref()
            .and_then(parse_date_stamp)
            .map(|date| date.to_string());
        let description = caps.get(2).map(|value| value.as_str().replace('_', " "));
        let version = caps.get(3).map(|value| value.as_str().to_lowercase());
        let status = caps.get(4).map(|value| value.as_str().to_uppercase());
        (date_stamp, parsed_date, description, version, status)
    } else {
        (None, None, None, None, None)
    };
    Ok(ScannedFilePayload {
        name,
        path: path.to_string_lossy().to_string(),
        size: metadata.len(),
        ext,
        doc_type: None,
        date_stamp,
        parsed_date,
        is_standard,
        description,
        version,
        status,
    })
}

fn parse_date_stamp(value: &str) -> Option<NaiveDate> {
    if value.len() != 6 {
        return None;
    }
    let year = value[0..2].parse::<i32>().ok()?;
    let month = value[2..4].parse::<u32>().ok()?;
    let day = value[4..6].parse::<u32>().ok()?;
    let full_year = if year <= 50 { 2000 + year } else { 1900 + year };
    NaiveDate::from_ymd_opt(full_year, month, day)
}

fn build_deadlines(appeal_date: Option<&str>, files: &[ScannedFilePayload]) -> Vec<DeadlineEventPayload> {
    let today = Local::now().date_naive();
    let mut events = Vec::new();
    if let Some(raw_date) = appeal_date.and_then(|value| NaiveDate::parse_from_str(value, "%Y-%m-%d").ok()) {
        events.push(to_deadline("appeal_date", "appeal_date", raw_date, "appeal.timeline.event.appealDate", None, None, today));
        events.push(to_deadline("response_due", "response_due", raw_date + chrono::Days::new(30), "appeal.timeline.event.responseDue", None, None, today));
    }
    for file in files {
        let Some(date) = file.parsed_date.as_deref().and_then(|value| NaiveDate::parse_from_str(value, "%Y-%m-%d").ok()) else {
            continue;
        };
        events.push(to_deadline(
            &format!("file_{}", file.path),
            "file_date",
            date,
            "appeal.timeline.event.fileDate",
            Some(file.path.clone()),
            Some(file.name.clone()),
            today,
        ));
    }
    events.sort_by(|left, right| left.date.cmp(&right.date));
    events
}

fn to_deadline(
    id: &str,
    kind: &str,
    date: NaiveDate,
    label_key: &str,
    file_path: Option<String>,
    file_name: Option<String>,
    today: NaiveDate,
) -> DeadlineEventPayload {
    let days = date.signed_duration_since(today).num_days();
    DeadlineEventPayload {
        id: id.to_string(),
        date: date.to_string(),
        label_key: label_key.to_string(),
        kind: kind.to_string(),
        is_past: days < 0,
        is_today: days == 0,
        is_urgent: (0..=3).contains(&days),
        file_path,
        file_name,
    }
}

fn compute_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn extract_text(path: &Path) -> AppResult<String> {
    let extension = path.extension().and_then(|value| value.to_str()).unwrap_or_default().to_lowercase();
    match extension.as_str() {
        "txt" | "md" | "markdown" => fs::read_to_string(path).map_err(AppError::from),
        "docx" => extract_docx(path),
        "pdf" => pdf_extract::extract_text(path).map_err(|err| AppError::new("errors.document.extractFailed", err.to_string())),
        _ => Ok(String::new()),
    }
}

fn extract_docx(path: &Path) -> AppResult<String> {
    let file = fs::File::open(path)?;
    let mut archive = ZipArchive::new(file)
        .map_err(|err: zip::result::ZipError| AppError::new("errors.document.extractFailed", err.to_string()))?;
    let mut xml = String::new();
    archive
        .by_name("word/document.xml")
        .map_err(|err: zip::result::ZipError| AppError::new("errors.document.extractFailed", err.to_string()))?
        .read_to_string(&mut xml)
        .map_err(|err: std::io::Error| AppError::new("errors.document.extractFailed", err.to_string()))?;
    let text = Regex::new(r"<[^>]+>")
        .map_err(|err: regex::Error| AppError::new("errors.regex.invalid", err.to_string()))?
        .replace_all(&xml, " ")
        .to_string();
    Ok(text)
}
