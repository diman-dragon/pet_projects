mod app_builder;
pub mod db;
pub mod commands;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    app_builder::build()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
