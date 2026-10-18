mod commands;
mod error;
mod models;
mod rag_commands;
mod license_commands;
mod repository;
mod services;
mod sqlite_vec_commands;

// New services
mod model_downloader;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(commands::db_commands::DbConnectionStore::default())
        .setup(|_app| {
            // Initialize reranker model
            if let Ok(path) = crate::model_downloader::ensure_reranker_model() {
                let path_str: &str = path.to_str().unwrap_or_default();
                crate::rag_commands::onnx::cross_encoder::CrossEncoder::init(path_str).ok();
            }

            // Регистрируем sqlite-vec глобально ДО открытия любых соединений.
            // После этого vec0 доступен во всех sqlite-соединениях процесса.
            sqlite_vec_commands::register_sqlite_vec();
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::business_commands::list_clients,
            commands::business_commands::save_client,
            commands::business_commands::delete_client,
            commands::business_commands::get_client_details,
            commands::business_commands::get_client_dossier,
            commands::business_commands::refresh_client_dossier,
            commands::business_commands::get_hierarchy_tree,
            commands::business_commands::save_case,
            commands::business_commands::delete_case,
            commands::business_commands::get_case_details,
            commands::business_commands::save_line,
            commands::business_commands::delete_line,
            commands::business_commands::get_line_details,
            commands::business_commands::save_appeal,
            commands::business_commands::list_agency_dictionary,
            commands::business_commands::suggest_agency_code,
            commands::business_commands::save_agency_dictionary_entry,
            commands::business_commands::delete_appeal,
            commands::business_commands::get_appeal_details,
            commands::business_commands::get_appeal_workbench,
            commands::business_commands::rename_appeal_file,
            commands::business_commands::upload_document,
            commands::business_commands::upload_appeal_document,
            commands::business_commands::retro_process_appeal_documents,
            commands::business_commands::reanalyze_document,
            commands::business_commands::generate_client_summary,
            commands::business_commands::generate_case_summary,
            commands::business_commands::generate_line_summary,
            commands::business_commands::get_dashboard_stats,
            commands::business_commands::list_trash_items,
            commands::business_commands::restore_trash_item,
            commands::business_commands::purge_trash_item,
            sqlite_vec_commands::check_sqlite_vec,
            license_commands::get_machine_fingerprint,
            license_commands::verify_master_key,
            license_commands::verify_license_key,
            rag_commands::chunking::rag_chunk_text,
            rag_commands::legal_chunker::rag_chunk_legal,
            rag_commands::math::rag_cosine_similarity,
            rag_commands::hybrid::rag_merge_hybrid,
            rag_commands::reranker::rag_rerank_candidates,
            rag_commands::math::rag_normalize,
            rag_commands::math::rag_score_vectors,
            // Document Commands
            commands::document_commands::extract_document_text,
            // AI Commands
            commands::ai_commands::ai_start_sidecar,
            commands::ai_commands::ai_stop_sidecar,
            commands::ai_commands::ai_get_status,
            commands::ai_commands::ai_get_model_library,
            commands::ai_commands::ai_list_models,
            commands::ai_commands::ai_scan_models,
            commands::ai_commands::ai_load_model,
            commands::ai_commands::ai_unload_model,
            commands::ai_commands::ai_chat_completion,
            commands::ai_commands::ai_pull_model,
            commands::ai_commands::ai_get_hardware_info,
            commands::ai_commands::ai_install_ollama,
            commands::ai_commands::ai_send_heartbeat,
            // [5.1] Прогресс скачивания модели
            commands::ai_pull_progress::ai_pull_model_with_progress,
            commands::db_commands::db_initialize,
            commands::db_commands::db_execute,
            commands::db_commands::db_select,
            commands::db_commands::db_close,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
