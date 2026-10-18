use tauri::AppHandle;

use crate::{
    error::{AppError, AppResult},
    models::domain::HierarchyClientNode,
    repository::{db, hierarchy_repository},
};

pub fn get_hierarchy_tree(app: &AppHandle) -> AppResult<Vec<HierarchyClientNode>> {
    let conn = db::open_clients_db(app)?;
    hierarchy_repository::load_tree(&conn).map_err(AppError::from)
}
