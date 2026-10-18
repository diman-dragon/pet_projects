use tauri::AppHandle;

use crate::{
    error::{AppError, AppResult},
    models::domain::DashboardStats,
    repository::{dashboard_repository, db},
};

pub fn load_dashboard_stats(app: &AppHandle) -> AppResult<DashboardStats> {
    let clients_conn = db::open_clients_db(app)?;
    let user_conn = db::open_user_db(app)?;
    Ok(DashboardStats {
        docs_count: dashboard_repository::count_table(&clients_conn, "cases")?,
        clients_count: dashboard_repository::count_table(&clients_conn, "clients")?,
        win_rate: dashboard_repository::count_table(&clients_conn, "lines")?,
        pnp_progress: dashboard_repository::count_table(&clients_conn, "appeals")?,
        client_dist: dashboard_repository::client_distribution(&clients_conn)?,
        cases_flow: dashboard_repository::case_flow(&clients_conn)?,
        deadlines: dashboard_repository::deadlines(&clients_conn)?,
        recent_logs: dashboard_repository::recent_logs(&user_conn, 10).map_err(AppError::from)?,
    })
}
