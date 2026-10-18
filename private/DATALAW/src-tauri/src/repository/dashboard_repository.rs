use rusqlite::{params, Connection};

use crate::models::domain::{
    DashboardCaseFlowItem, DashboardClientDistributionItem, DashboardDeadlineItem, DashboardLogItem,
};

pub fn count_table(conn: &Connection, table: &str) -> rusqlite::Result<i64> {
    conn.query_row(
        &format!("SELECT COUNT(*) FROM {table} WHERE deleted_at IS NULL"),
        [],
        |row| row.get(0),
    )
}

pub fn client_distribution(conn: &Connection) -> rusqlite::Result<Vec<DashboardClientDistributionItem>> {
    let mut stmt = conn.prepare(
        "SELECT COALESCE(type, 'UNKNOWN'), COUNT(*)
         FROM clients
         WHERE deleted_at IS NULL
         GROUP BY type
         ORDER BY COUNT(*) DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        let raw_type: String = row.get(0)?;
        let (name, color) = match raw_type.as_str() {
            "LEGAL" | "LE" => ("Юрлица".to_string(), "#0067c0".to_string()),
            "INDIVIDUAL" | "IP" => ("Физлица".to_string(), "#10b981".to_string()),
            _ => ("Прочее".to_string(), "#94a3b8".to_string()),
        };
        Ok(DashboardClientDistributionItem {
            name,
            value: row.get(1)?,
            color,
        })
    })?;
    rows.collect()
}

pub fn recent_logs(conn: &Connection, limit: i64) -> rusqlite::Result<Vec<DashboardLogItem>> {
    let mut stmt = conn.prepare(
        "SELECT id, created_at, COALESCE(actor_login, 'system'), action, entity_type, details
         FROM audit_log
         ORDER BY created_at DESC, id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit], |row| {
        let action: String = row.get(3)?;
        let entity_type: String = row.get(4)?;
        let fallback = format!("{action} {entity_type}");
        Ok(DashboardLogItem {
            id: row.get(0)?,
            time: row.get(1)?,
            user: row.get(2)?,
            msg: row.get::<_, Option<String>>(5)?.unwrap_or(fallback),
        })
    })?;
    rows.collect()
}

pub fn case_flow(conn: &Connection) -> rusqlite::Result<Vec<DashboardCaseFlowItem>> {
    let mut stmt = conn.prepare(
        "SELECT
            strftime('%Y-%m', ca.created_at) AS month,
            SUM(CASE WHEN COALESCE(cl.type, '') IN ('LEGAL', 'LE') THEN 1 ELSE 0 END) AS corporate,
            SUM(CASE WHEN COALESCE(cl.type, '') IN ('INDIVIDUAL', 'IP') THEN 1 ELSE 0 END) AS private
         FROM cases ca
         JOIN clients cl ON cl.id = ca.client_id
         WHERE ca.deleted_at IS NULL AND cl.deleted_at IS NULL AND ca.created_at IS NOT NULL
         GROUP BY strftime('%Y-%m', ca.created_at)
         ORDER BY month DESC
         LIMIT 6",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(DashboardCaseFlowItem {
            month: row.get::<_, Option<String>>(0)?.unwrap_or_default(),
            corporate: row.get::<_, Option<i64>>(1)?.unwrap_or(0),
            private: row.get::<_, Option<i64>>(2)?.unwrap_or(0),
        })
    })?;
    rows.collect()
}

pub fn deadlines(_conn: &Connection) -> rusqlite::Result<Vec<DashboardDeadlineItem>> {
    Ok(Vec::new())
}
