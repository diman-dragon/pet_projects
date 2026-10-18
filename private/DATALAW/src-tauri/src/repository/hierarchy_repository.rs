use std::collections::BTreeMap;

use rusqlite::Connection;

use crate::models::domain::{
    HierarchyAppealNode, HierarchyCaseNode, HierarchyClientNode, HierarchyLineNode,
};

pub fn load_tree(conn: &Connection) -> rusqlite::Result<Vec<HierarchyClientNode>> {
    let mut stmt = conn.prepare(
        "SELECT
            cl.id,
            cl.full_name,
            cl.folder_name,
            ca.id,
            ca.title,
            ca.folder_name,
            ca.status,
            ca.case_type,
            ca.counterparty,
            ca.litigation_subtype,
            l.id,
            l.title,
            l.folder_name,
            a.id,
            a.title,
            a.folder_name
         FROM clients cl
         LEFT JOIN cases ca ON ca.client_id = cl.id AND ca.deleted_at IS NULL
         LEFT JOIN lines l ON l.case_id = ca.id AND l.deleted_at IS NULL
         LEFT JOIN appeals a ON a.line_id = l.id AND a.deleted_at IS NULL
         WHERE cl.deleted_at IS NULL
         ORDER BY cl.full_name ASC, ca.title ASC, l.id ASC, a.created_at DESC, a.id DESC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<i64>>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, Option<String>>(6)?,
            row.get::<_, Option<String>>(7)?,
            row.get::<_, Option<String>>(8)?,
            row.get::<_, Option<String>>(9)?,
            row.get::<_, Option<i64>>(10)?,
            row.get::<_, Option<String>>(11)?,
            row.get::<_, Option<String>>(12)?,
            row.get::<_, Option<i64>>(13)?,
            row.get::<_, Option<String>>(14)?,
            row.get::<_, Option<String>>(15)?,
        ))
    })?;

    let mut clients = BTreeMap::<i64, HierarchyClientNode>::new();
    for row in rows {
        let (client_id, full_name, client_folder, case_id, case_title, case_folder, case_status, case_type, counterparty, litigation_subtype, line_id, line_title, line_folder, appeal_id, appeal_title, appeal_folder) = row?;
        let client = clients.entry(client_id).or_insert_with(|| HierarchyClientNode {
            id: client_id,
            full_name,
            folder_name: client_folder,
            cases: Vec::new(),
        });
        let Some(case_id) = case_id else {
            continue;
        };
        let case_index = ensure_case(
            client,
            case_id,
            case_title.unwrap_or_default(),
            case_folder.unwrap_or_default(),
            case_status.unwrap_or_default(),
            case_type,
            counterparty,
            litigation_subtype,
        );
        let Some(line_id) = line_id else {
            continue;
        };
        let line_index = ensure_line(
            &mut client.cases[case_index],
            line_id,
            line_title.unwrap_or_default(),
            line_folder.unwrap_or_default(),
        );
        if let Some(appeal_id) = appeal_id {
            client.cases[case_index].lines[line_index]
                .appeals
                .push(HierarchyAppealNode {
                    id: appeal_id,
                    title: appeal_title.unwrap_or_default(),
                    folder_name: appeal_folder.unwrap_or_default(),
                });
        }
    }
    Ok(clients.into_values().collect())
}

fn ensure_case(client: &mut HierarchyClientNode, case_id: i64, title: String, folder_name: String, status: String, case_type: Option<String>, counterparty: Option<String>, litigation_subtype: Option<String>) -> usize {
    if let Some(index) = client.cases.iter().position(|item| item.id == case_id) {
        return index;
    }
    client.cases.push(HierarchyCaseNode {
        id: case_id,
        title,
        folder_name,
        status,
        case_type,
        counterparty,
        litigation_subtype,
        lines: Vec::new(),
    });
    client.cases.len() - 1
}

fn ensure_line(case_item: &mut HierarchyCaseNode, line_id: i64, title: String, folder_name: String) -> usize {
    if let Some(index) = case_item.lines.iter().position(|item| item.id == line_id) {
        return index;
    }
    case_item.lines.push(HierarchyLineNode {
        id: line_id,
        title,
        folder_name,
        appeals: Vec::new(),
    });
    case_item.lines.len() - 1
}
