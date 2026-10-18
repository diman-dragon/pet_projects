// src-tauri/src/main.rs

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // Обращаемся к имени, которое мы дали в [lib] в Cargo.toml
    Dorlane_app_lib::run();
}
