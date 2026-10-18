use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::database::DatabaseState;
use crate::geometry::PERSON_CARD_WIDTH;
pub use crate::transfer_types::ImportResult;
use crate::transfer_types::{TransferPerson, TransferRelation};

const CARD_GAP: f64 = 100.0;
const CARD_STEP: f64 = PERSON_CARD_WIDTH + CARD_GAP;

#[derive(Debug, Serialize, Deserialize)]
struct TransferBundle {
    people: Vec<TransferPerson>,
    #[serde(default)]
    relations: Vec<TransferRelation>,
}

fn open_database(state: &DatabaseState) -> Result<Connection, String> {
    let path = state
        .path
        .lock()
        .map_err(|_| "Не удалось получить путь базы".to_owned())?
        .clone();
    let connection = Connection::open(path).map_err(|error| error.to_string())?;
    connection
        .execute_batch("PRAGMA foreign_keys = ON;")
        .map_err(|error| error.to_string())?;
    Ok(connection)
}

fn atomic_write(path: &Path, content: &str) -> Result<(), String> {
    let suffix = Uuid::new_v4().simple().to_string();
    let temporary = path.with_file_name(format!(
        ".{}.export-{suffix}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("family-tree")
    ));
    fs::write(&temporary, content).map_err(|error| error.to_string())?;
    let backup = path.with_file_name(format!(
        ".{}.export-backup-{suffix}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("family-tree")
    ));
    let had_existing = path.exists();
    if had_existing {
        if let Err(error) = fs::rename(path, &backup) {
            let _ = fs::remove_file(&temporary);
            return Err(error.to_string());
        }
    }
    if let Err(error) = fs::rename(&temporary, path) {
        if had_existing {
            let _ = fs::rename(&backup, path);
        }
        let _ = fs::remove_file(&temporary);
        return Err(error.to_string());
    }
    let _ = fs::remove_file(&backup);
    Ok(())
}

fn read_people(connection: &Connection) -> Result<Vec<TransferPerson>, String> {
    let mut statement = connection.prepare("SELECT c.id,c.x,c.y,c.generation,p.sex,p.photo_url,p.gender_identity,p.nationality,p.ethnicity,p.religion,p.occupation,p.education,p.description,p.biography,p.research_status,(SELECT pn.given_name FROM person_names pn WHERE pn.person_id=p.id AND pn.is_preferred=1 LIMIT 1),(SELECT pn.patronymic FROM person_names pn WHERE pn.person_id=p.id AND pn.is_preferred=1 LIMIT 1),(SELECT pn.surname FROM person_names pn WHERE pn.person_id=p.id AND pn.is_preferred=1 LIMIT 1),(SELECT pn.maiden_name FROM person_names pn WHERE pn.person_id=p.id AND pn.is_preferred=1 LIMIT 1),(SELECT dv.original_text FROM events e JOIN event_participants ep ON ep.event_id=e.id JOIN date_values dv ON dv.id=e.date_id WHERE ep.person_id=p.id AND e.event_type='birth' LIMIT 1),(SELECT dv.original_text FROM events e JOIN event_participants ep ON ep.event_id=e.id JOIN date_values dv ON dv.id=e.date_id WHERE ep.person_id=p.id AND e.event_type='death' LIMIT 1) FROM canvas_cards c JOIN persons p ON p.id=c.person_id WHERE p.deleted_at IS NULL ORDER BY c.created_at").map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok(TransferPerson {
                id: row.get(0)?,
                x: Some(row.get(1)?),
                y: Some(row.get(2)?),
                generation: Some(row.get(3)?),
                sex: row.get(4)?,
                photo_url: row.get(5)?,
                gender_identity: row.get(6)?,
                nationality: row.get(7)?,
                ethnicity: row.get(8)?,
                religion: row.get(9)?,
                occupation: row.get(10)?,
                education: row.get(11)?,
                description: row.get(12)?,
                biography: row.get(13)?,
                research_status: row.get(14)?,
                first_name: row.get(15)?,
                patronymic: row.get(16)?,
                last_name: row.get(17)?,
                previous_last_name: row.get(18)?,
                birth_date: row.get(19)?,
                death_date: row.get(20)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|error| error.to_string())
}

fn read_relations(connection: &Connection) -> Result<Vec<TransferRelation>, String> {
    let mut statement = connection
        .prepare("SELECT u.id,u.union_type,a.person_id,b.person_id,u.details_json FROM unions u JOIN union_members a ON a.union_id=u.id AND a.sequence=1 JOIN union_members b ON b.union_id=u.id AND b.sequence=2 ORDER BY u.id")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok(TransferRelation {
                id: row.get(0)?,
                relation_type: row.get(1)?,
                person_a_id: row.get(2)?,
                person_b_id: row.get(3)?,
                details_json: row.get(4)?,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|error| error.to_string())
}

fn csv_escape(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn people_csv(people: &[TransferPerson]) -> String {
    let mut output = String::from("id,first_name,patronymic,last_name,previous_last_name,sex,birth_date,death_date,photo_url,x,y,generation,gender_identity,nationality,ethnicity,religion,occupation,education,description,biography,research_status\n");
    for person in people {
        let values = [
            person.id.clone(),
            person.first_name.clone().unwrap_or_default(),
            person.patronymic.clone().unwrap_or_default(),
            person.last_name.clone().unwrap_or_default(),
            person.previous_last_name.clone().unwrap_or_default(),
            person.sex.clone().unwrap_or_default(),
            person.birth_date.clone().unwrap_or_default(),
            person.death_date.clone().unwrap_or_default(),
            person.photo_url.clone().unwrap_or_default(),
            person.x.unwrap_or_default().to_string(),
            person.y.unwrap_or_default().to_string(),
            person.generation.unwrap_or_default().to_string(),
            person.gender_identity.clone().unwrap_or_default(),
            person.nationality.clone().unwrap_or_default(),
            person.ethnicity.clone().unwrap_or_default(),
            person.religion.clone().unwrap_or_default(),
            person.occupation.clone().unwrap_or_default(),
            person.education.clone().unwrap_or_default(),
            person.description.clone().unwrap_or_default(),
            person.biography.clone().unwrap_or_default(),
            person.research_status.clone().unwrap_or_default(),
        ];
        output.push_str(
            &values
                .iter()
                .map(|value| csv_escape(value))
                .collect::<Vec<_>>()
                .join(","),
        );
        output.push('\n');
    }
    output
}

fn csv_fields(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for character in line.chars() {
        match character {
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(current.clone());
                current.clear();
            }
            _ => current.push(character),
        }
    }
    fields.push(current);
    fields
        .into_iter()
        .map(|value| value.trim_matches('"').replace("\"\"", "\""))
        .collect()
}

fn optional_value(value: &str) -> Option<String> {
    (!value.trim().is_empty()).then_some(value.trim().to_owned())
}

fn parse_csv(content: &str) -> Vec<TransferPerson> {
    content
        .lines()
        .skip(1)
        .filter_map(|line| {
            let fields = csv_fields(line);
            if fields.len() < 12 {
                return None;
            }
            Some(TransferPerson {
                id: fields[0].clone(),
                first_name: optional_value(&fields[1]),
                patronymic: optional_value(&fields[2]),
                last_name: optional_value(&fields[3]),
                previous_last_name: optional_value(&fields[4]),
                sex: optional_value(&fields[5]),
                birth_date: optional_value(&fields[6]),
                death_date: optional_value(&fields[7]),
                photo_url: optional_value(&fields[8]),
                x: fields[9].parse().ok(),
                y: fields[10].parse().ok(),
                generation: fields[11].parse().ok(),
                gender_identity: fields.get(12).and_then(|value| optional_value(value)),
                nationality: fields.get(13).and_then(|value| optional_value(value)),
                ethnicity: fields.get(14).and_then(|value| optional_value(value)),
                religion: fields.get(15).and_then(|value| optional_value(value)),
                occupation: fields.get(16).and_then(|value| optional_value(value)),
                education: fields.get(17).and_then(|value| optional_value(value)),
                description: fields.get(18).and_then(|value| optional_value(value)),
                biography: fields.get(19).and_then(|value| optional_value(value)),
                research_status: fields.get(20).and_then(|value| optional_value(value)),
            })
        })
        .collect()
}

fn parse_gedcom(content: &str) -> Vec<TransferPerson> {
    let mut people = Vec::new();
    let mut current: Option<TransferPerson> = None;
    let mut event: Option<&str> = None;
    for line in content.lines() {
        let mut parts = line.splitn(3, ' ');
        let level = parts.next().unwrap_or_default();
        let tag = parts.next().unwrap_or_default();
        let value = parts.next().unwrap_or_default().trim();
        if level == "0" && tag.starts_with('@') && tag.ends_with('@') && value == "INDI" {
            if let Some(person) = current.take() {
                people.push(person);
            }
            current = Some(TransferPerson {
                id: tag.trim_matches('@').to_owned(),
                first_name: None,
                patronymic: None,
                last_name: None,
                previous_last_name: None,
                sex: None,
                birth_date: None,
                death_date: None,
                photo_url: None,
                x: None,
                y: None,
                generation: None,
                gender_identity: None,
                nationality: None,
                ethnicity: None,
                religion: None,
                occupation: None,
                education: None,
                description: None,
                biography: None,
                research_status: None,
            });
            event = None;
            continue;
        }
        let Some(person) = current.as_mut() else {
            continue;
        };
        match (level, tag) {
            ("1", "NAME") => {
                person.first_name = value
                    .replace('/', "")
                    .split_whitespace()
                    .next()
                    .map(str::to_owned);
                person.last_name = value.split('/').nth(1).map(str::to_owned);
            }
            ("2", "GIVN") => person.first_name = optional_value(value),
            ("2", "SURN") => person.last_name = optional_value(value),
            ("1", "SEX") => {
                person.sex = Some(if value == "F" {
                    "female".to_owned()
                } else {
                    "male".to_owned()
                })
            }
            ("1", "BIRT") => event = Some("birth"),
            ("1", "DEAT") => event = Some("death"),
            ("2", "DATE") if event == Some("birth") => person.birth_date = optional_value(value),
            ("2", "DATE") if event == Some("death") => person.death_date = optional_value(value),
            _ => {}
        }
    }
    if let Some(person) = current {
        people.push(person);
    }
    people
}

fn gedcom(people: &[TransferPerson], version: &str) -> String {
    let mut output = format!("0 HEAD\n1 GEDC\n2 VERS {version}\n1 CHAR UTF-8\n");
    for person in people {
        let name = [
            person.first_name.clone(),
            person.patronymic.clone(),
            person.last_name.clone(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
        output.push_str(&format!(
            "0 @{}@ INDI\n1 NAME {} /{}/\n",
            person.id,
            name,
            person.last_name.clone().unwrap_or_default()
        ));
        if let Some(sex) = &person.sex {
            output.push_str(&format!(
                "1 SEX {}\n",
                if sex == "female" { "F" } else { "M" }
            ));
        }
        if let Some(date) = &person.birth_date {
            output.push_str(&format!("1 BIRT\n2 DATE {date}\n"));
        }
        if let Some(date) = &person.death_date {
            output.push_str(&format!("1 DEAT\n2 DATE {date}\n"));
        }
    }
    output.push_str("0 TRLR\n");
    output
}

fn generated_id(connection: &Connection) -> Result<String, String> {
    let value: i64 = connection
        .query_row(
            "SELECT next_value FROM id_sequences WHERE name='person'",
            [],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    if value > 9999 {
        return Err("Достигнут предел идентификаторов P9999".to_owned());
    }
    connection
        .execute(
            "UPDATE id_sequences SET next_value=next_value+1 WHERE name='person'",
            [],
        )
        .map_err(|error| error.to_string())?;
    Ok(format!("P{value:04}"))
}

fn duplicate_key(person: &TransferPerson) -> String {
    [
        person.first_name.as_deref().unwrap_or_default(),
        person.patronymic.as_deref().unwrap_or_default(),
        person.last_name.as_deref().unwrap_or_default(),
        person.birth_date.as_deref().unwrap_or_default(),
        person.death_date.as_deref().unwrap_or_default(),
    ]
    .join("|")
    .split_whitespace()
    .collect::<Vec<_>>()
    .join(" ")
    .to_lowercase()
}

fn import_people(
    connection: &mut Connection,
    people: &[TransferPerson],
    replace: bool,
    generation_height: f64,
) -> Result<(ImportResult, HashMap<String, String>), String> {
    let mut existing_keys = HashSet::new();
    let mut existing_ids = HashMap::new();
    if !replace {
        let existing = read_people(connection)?;
        for person in existing {
            let key = duplicate_key(&person);
            existing_ids.insert(key.clone(), person.id);
            existing_keys.insert(key);
        }
    }
    let mut occupied: Vec<(i32, f64)> = if replace {
        Vec::new()
    } else {
        let mut statement = connection
            .prepare("SELECT generation, x FROM canvas_cards c JOIN persons p ON p.id = c.person_id WHERE p.deleted_at IS NULL")
            .map_err(|error| error.to_string())?;
        let result = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .map_err(|error| error.to_string())?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|error| error.to_string())?;
        result
    };
    let max_existing_generation = occupied
        .iter()
        .map(|(generation, _)| *generation)
        .max()
        .unwrap_or(-1);
    let fallback_generation = max_existing_generation + 1;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    if replace {
        transaction
            .execute("DELETE FROM persons", [])
            .map_err(|error| error.to_string())?;
        transaction
            .execute(
                "UPDATE id_sequences SET next_value=1 WHERE name='person'",
                [],
            )
            .map_err(|error| error.to_string())?;
    }
    let mut added = 0;
    let mut skipped_duplicates = 0;
    let mut id_map = HashMap::new();
    for person in people {
        if !replace && !existing_keys.insert(duplicate_key(person)) {
            skipped_duplicates += 1;
            if let Some(existing_id) = existing_ids.get(&duplicate_key(person)) {
                id_map.insert(person.id.clone(), existing_id.clone());
            }
            continue;
        }
        let id = generated_id(&transaction)?;
        let generation = person
            .generation
            .or_else(|| {
                person
                    .y
                    .map(|value| (value / generation_height).round() as i32)
            })
            .unwrap_or(fallback_generation);
        let mut x = person.x.unwrap_or(0.0);
        while occupied.iter().any(|(existing_generation, existing_x)| {
            *existing_generation == generation && (x - existing_x).abs() < CARD_STEP
        }) {
            x += CARD_STEP;
        }
        let y = generation as f64 * generation_height;
        occupied.push((generation, x));
        transaction
            .execute(
            "INSERT INTO persons (id,sex,photo_url,gender_identity,nationality,ethnicity,religion,occupation,education,description,biography,research_status) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
                params![id, person.sex, person.photo_url, person.gender_identity, person.nationality, person.ethnicity, person.religion, person.occupation, person.education, person.description, person.biography, person.research_status],
            )
            .map_err(|error| error.to_string())?;
        id_map.insert(person.id.clone(), id.clone());
        transaction
            .execute(
                "INSERT INTO canvas_cards (id,person_id,x,y,generation) VALUES (?1,?1,?2,?3,?4)",
                params![id, x, y, generation],
            )
            .map_err(|error| error.to_string())?;
        let full_name = [
            person.first_name.clone(),
            person.patronymic.clone(),
            person.last_name.clone(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
        if !full_name.is_empty() || person.previous_last_name.is_some() {
            transaction.execute("INSERT INTO person_names (id,person_id,full_text,given_name,patronymic,surname,maiden_name,is_preferred) VALUES (?1,?2,?3,?4,?5,?6,?7,1)", params![Uuid::new_v4().to_string(),id,full_name,person.first_name,person.patronymic,person.last_name,person.previous_last_name]).map_err(|error| error.to_string())?;
        }
        added += 1;
    }
    transaction.commit().map_err(|error| error.to_string())?;
    Ok((
        ImportResult {
            added,
            skipped_duplicates,
            replaced: replace,
            relations_added: 0,
        },
        id_map,
    ))
}

fn import_relations(
    connection: &mut Connection,
    relations: &[TransferRelation],
    id_map: &HashMap<String, String>,
) -> Result<usize, String> {
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    let mut added = 0;
    for relation in relations {
        if !matches!(relation.relation_type.as_str(), "marriage" | "parenthood") {
            continue;
        }
        let Some(person_a_id) = id_map.get(&relation.person_a_id) else {
            continue;
        };
        let Some(person_b_id) = id_map.get(&relation.person_b_id) else {
            continue;
        };
        if person_a_id == person_b_id {
            continue;
        }
        let duplicate: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM unions u JOIN union_members a ON a.union_id=u.id JOIN union_members b ON b.union_id=u.id WHERE u.union_type=?3 AND ((a.person_id=?1 AND b.person_id=?2) OR (a.person_id=?2 AND b.person_id=?1))",
            params![person_a_id, person_b_id, relation.relation_type],
            |row| row.get(0),
        ).map_err(|error| error.to_string())?;
        if duplicate > 0 {
            continue;
        }
        let union_id = Uuid::new_v4().to_string();
        transaction
            .execute(
                "INSERT INTO unions (id,union_type,details_json) VALUES (?1,?2,?3)",
                params![union_id, relation.relation_type, relation.details_json],
            )
            .map_err(|error| error.to_string())?;
        let (role_a, role_b) = if relation.relation_type == "parenthood" {
            ("parent", "child")
        } else {
            ("spouse", "spouse")
        };
        transaction.execute("INSERT INTO union_members (union_id,person_id,role,sequence) VALUES (?1,?2,?3,1),(?1,?4,?5,2)", params![union_id, person_a_id, role_a, person_b_id, role_b]).map_err(|error| error.to_string())?;
        added += 1;
    }
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(added)
}

#[tauri::command]
pub fn export_data(
    path: String,
    format: String,
    state: tauri::State<'_, DatabaseState>,
) -> Result<(), String> {
    let connection = open_database(&state)?;
    let people = read_people(&connection)?;
    let content = match format.as_str() {
        "json" => serde_json::to_string_pretty(&TransferBundle {
            people,
            relations: read_relations(&connection)?,
        })
        .map_err(|error| error.to_string())?,
        "csv" => people_csv(&people),
        "gedcom7" => gedcom(&people, "7.0"),
        "gedcom551" => gedcom(&people, "5.5.1"),
        _ => return Err("Неподдерживаемый формат экспорта".to_owned()),
    };
    atomic_write(Path::new(&path), &content)
}

#[tauri::command]
pub fn import_data(
    path: String,
    format: String,
    mode: String,
    generation_height: f64,
    state: tauri::State<'_, DatabaseState>,
) -> Result<ImportResult, String> {
    if mode != "merge" && mode != "replace" {
        return Err("Неизвестный режим импорта: выберите «Объединить» или «Заменить».".to_owned());
    }
    let input_path = PathBuf::from(path);
    let metadata = fs::metadata(&input_path).map_err(|error| error.to_string())?;
    if metadata.len() > 100 * 1024 * 1024 {
        return Err("Файл импорта слишком большой: максимальный размер — 100 МБ.".to_owned());
    }
    let content = fs::read_to_string(input_path).map_err(|error| error.to_string())?;
    let (people, relations): (Vec<TransferPerson>, Vec<TransferRelation>) = match format.as_str() {
        "json" => {
            let bundle = serde_json::from_str::<TransferBundle>(&content)
                .map_err(|error| format!("Некорректный JSON-пакет дерева: {error}"))?;
            (bundle.people, bundle.relations)
        }
        "csv" => (parse_csv(&content), Vec::new()),
        "gedcom7" | "gedcom551" => (parse_gedcom(&content), Vec::new()),
        _ => return Err("Неподдерживаемый формат импорта".to_owned()),
    };
    let mut connection = open_database(&state)?;
    if !generation_height.is_finite() || generation_height <= 0.0 {
        return Err("Некорректная высота поколения".to_owned());
    }
    let (mut result, id_map) = import_people(
        &mut connection,
        &people,
        mode == "replace",
        generation_height,
    )?;
    result.relations_added = import_relations(&mut connection, &relations, &id_map)?;
    if result.added == 0 && !people.is_empty() && !result.replaced {
        return Err("Импорт не добавил записей: все записи уже существуют или не прошли проверку дубликатов.".to_owned());
    }
    Ok(result)
}
