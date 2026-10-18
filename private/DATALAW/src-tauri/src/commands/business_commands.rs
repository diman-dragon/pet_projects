use tauri::AppHandle;

use crate::{
    error::AppResult,
    models::payloads::{
        AppealDetailsPayload, AppealWorkbenchPayload, CaseDetailsPayload, ClientDetailsPayload, ClientDossierOverviewPayload,
        LineDetailsPayload, RefreshClientDossierPayload, RefreshClientDossierRequest, SaveAgencyDictionaryPayload,
        SaveAppealPayload, SaveCasePayload, SaveClientPayload, SaveLinePayload, SummaryGenerationOptions, UploadDocumentPayload,
    },
    services::{agency_codes, appeal, case, client, client_dossier, dashboard, document, hierarchy, line, trash},
};

use crate::models::domain::{Client, DashboardStats, HierarchyClientNode, TrashItem};

#[tauri::command]
pub fn list_clients(app: AppHandle) -> AppResult<Vec<Client>> {
    client::list_clients(&app)
}

#[tauri::command]
pub fn save_client(app: AppHandle, payload: SaveClientPayload) -> AppResult<()> {
    client::save_client(&app, payload)
}

#[tauri::command]
pub fn delete_client(app: AppHandle, client_id: i64) -> AppResult<()> {
    client::delete_client(&app, client_id)
}

#[tauri::command]
pub fn get_hierarchy_tree(app: AppHandle) -> AppResult<Vec<HierarchyClientNode>> {
    hierarchy::get_hierarchy_tree(&app)
}

#[tauri::command]
pub fn get_dashboard_stats(app: AppHandle) -> AppResult<DashboardStats> {
    dashboard::load_dashboard_stats(&app)
}

#[tauri::command]
pub fn list_trash_items(app: AppHandle) -> AppResult<Vec<TrashItem>> {
    trash::list_items(&app)
}

#[tauri::command]
pub fn restore_trash_item(app: AppHandle, entity_type: String, entity_id: i64) -> AppResult<()> {
    trash::restore_item(&app, &entity_type, entity_id)
}

#[tauri::command]
pub fn purge_trash_item(app: AppHandle, entity_type: String, entity_id: i64) -> AppResult<()> {
    trash::purge_item(&app, &entity_type, entity_id)
}

#[tauri::command]
pub fn get_client_details(app: AppHandle, client_id: i64) -> AppResult<ClientDetailsPayload> {
    case::load_client_details(&app, client_id)
}

#[tauri::command]
pub fn get_client_dossier(app: AppHandle, client_id: i64) -> AppResult<ClientDossierOverviewPayload> {
    client_dossier::get_client_dossier(&app, client_id)
}

#[tauri::command]
pub fn refresh_client_dossier(app: AppHandle, request: RefreshClientDossierRequest) -> AppResult<RefreshClientDossierPayload> {
    client_dossier::refresh_client_dossier(&app, request)
}

#[tauri::command]
pub fn save_case(app: AppHandle, payload: SaveCasePayload) -> AppResult<()> {
    case::save_case(&app, payload)
}

#[tauri::command]
pub fn delete_case(app: AppHandle, case_id: i64) -> AppResult<()> {
    case::delete_case(&app, case_id)
}

#[tauri::command]
pub fn get_case_details(app: AppHandle, case_id: i64) -> AppResult<CaseDetailsPayload> {
    line::load_case_details(&app, case_id)
}

#[tauri::command]
pub fn save_line(app: AppHandle, payload: SaveLinePayload) -> AppResult<()> {
    line::save_line(&app, payload)
}

#[tauri::command]
pub fn delete_line(app: AppHandle, line_id: i64) -> AppResult<()> {
    line::delete_line(&app, line_id)
}

#[tauri::command]
pub fn get_line_details(app: AppHandle, line_id: i64) -> AppResult<LineDetailsPayload> {
    line::load_line_details(&app, line_id)
}

#[tauri::command]
pub fn save_appeal(app: AppHandle, payload: SaveAppealPayload) -> AppResult<i64> {
    line::save_appeal(&app, payload)
}

#[tauri::command]
pub fn list_agency_dictionary(app: AppHandle) -> AppResult<Vec<agency_codes::AgencyDictionaryEntry>> {
    agency_codes::list_agency_codes(&app)
}

#[tauri::command]
pub fn suggest_agency_code(app: AppHandle, name: String) -> AppResult<agency_codes::AgencyCodeSuggestion> {
    agency_codes::suggest_agency_code(&app, &name)
}

#[tauri::command]
pub fn save_agency_dictionary_entry(
    app: AppHandle,
    payload: SaveAgencyDictionaryPayload,
) -> AppResult<agency_codes::AgencyDictionaryEntry> {
    agency_codes::save_agency_code(&app, &payload.name, &payload.code)
}

#[tauri::command]
pub fn delete_appeal(app: AppHandle, appeal_id: i64) -> AppResult<()> {
    line::delete_appeal(&app, appeal_id)
}

#[tauri::command]
pub fn get_appeal_details(app: AppHandle, appeal_id: i64) -> AppResult<AppealDetailsPayload> {
    appeal::load_appeal_details(&app, appeal_id)
}

#[tauri::command]
pub fn get_appeal_workbench(app: AppHandle, appeal_id: i64) -> AppResult<AppealWorkbenchPayload> {
    document::load_appeal_workbench(&app, appeal_id)
}

#[tauri::command]
pub fn rename_appeal_file(app: AppHandle, file_path: String, new_stem: String) -> AppResult<()> {
    document::rename_appeal_file(&app, &file_path, &new_stem)
}

#[tauri::command]
pub fn upload_document(app: AppHandle, payload: UploadDocumentPayload) -> AppResult<String> {
    document::upload_document(&app, payload)
}

#[tauri::command]
pub fn upload_appeal_document(app: AppHandle, appeal_id: i64, source_path: String) -> AppResult<AppealWorkbenchPayload> {
    document::upload_appeal_document(&app, appeal_id, &source_path)
}

#[tauri::command]
pub fn retro_process_appeal_documents(app: AppHandle, appeal_id: i64) -> AppResult<AppealWorkbenchPayload> {
    document::retro_process_appeal_documents(&app, appeal_id)
}

#[tauri::command]
pub fn reanalyze_document(app: AppHandle, appeal_id: i64, file_path: String) -> AppResult<AppealWorkbenchPayload> {
    document::reanalyze_document(&app, appeal_id, &file_path)
}

#[tauri::command]
pub fn generate_client_summary(
    app: AppHandle,
    client_id: i64,
    options: Option<SummaryGenerationOptions>,
) -> AppResult<()> {
    document::generate_client_summary(&app, client_id, options)
}

#[tauri::command]
pub fn generate_case_summary(
    app: AppHandle,
    case_id: i64,
    options: Option<SummaryGenerationOptions>,
) -> AppResult<()> {
    document::generate_case_summary(&app, case_id, options)
}

#[tauri::command]
pub fn generate_line_summary(
    app: AppHandle,
    line_id: i64,
    options: Option<SummaryGenerationOptions>,
) -> AppResult<()> {
    document::generate_line_summary(&app, line_id, options)
}
