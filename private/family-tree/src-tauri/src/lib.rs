mod database;
mod geometry;
mod transfer;
mod transfer_types;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let app_data_dir = app.path().app_data_dir()?;
            let config_path = app_data_dir.join("workspace.json");
            let current_database_path = database::saved_database_path(&config_path);
            let database_path = current_database_path.unwrap_or_else(|| {
                app_data_dir
                    .join("FamilyTree")
                    .join("Моё дерево")
                    .join("db")
                    .join("family-tree.sqlite3")
            });
            let media_dir = database_path
                .parent()
                .and_then(std::path::Path::parent)
                .ok_or("Не удалось определить каталог media")?
                .join("media");
            std::fs::create_dir_all(media_dir)?;
            database::initialize_database(&database_path)
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error>)?;
            app.manage(database::DatabaseState {
                path: std::sync::Mutex::new(database_path),
                config_path,
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            database::load_canvas_cards,
            database::load_relations,
            database::create_relation,
            database::update_relation,
            database::delete_relation,
            database::swap_relation_members,
            database::create_canvas_card,
            database::update_canvas_card_position,
            database::update_canvas_cards_positions,
            database::set_canvas_cards_locked,
            database::update_person,
            database::delete_canvas_card,
            database::clear_tree,
            database::clear_relations,
            database::restore_canvas_card,
            database::purge_canvas_card,
            database::purge_canvas_cards,
            database::load_trash_cards,
            database::get_workspace,
            database::set_workspace,
            database::import_person_media,
            database::read_local_file,
            database::save_person_photo_crop,
            database::delete_person_photo,
            database::save_relation_photo_crop,
            database::delete_relation_photo,
            database::update_relation_details,
            database::update_parenthood_auto_align_family,
            database::load_person_media,
            transfer::export_data,
            transfer::import_data,
        ])
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
