use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use serde_json::{json, Value};
use chrono::{Duration, Local};
use super::models::{GrammarUnitInput, GrammarExerciseInput};
use super::utils::normalize_tags;

pub fn get_grammar_dashboard(connection: &Connection) -> Result<Value, String> {
    connection
        .query_row(
            r#"
            SELECT
              COUNT(*),
              COALESCE(SUM(CASE WHEN status = 'not_started' THEN 1 ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN status = 'mastered' THEN 1 ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN status = 'mastered' AND next_review_at IS NOT NULL
                AND datetime(next_review_at) <= datetime('now', 'localtime') THEN 1 ELSE 0 END), 0)
            FROM grammar_units
            "#,
            [],
            |row| {
                Ok(json!({
                    "total":      row.get::<_, i64>(0)?,
                    "notStarted": row.get::<_, i64>(1)?,
                    "inProgress": row.get::<_, i64>(2)?,
                    "mastered":   row.get::<_, i64>(3)?,
                    "dueReviews": row.get::<_, i64>(4)?,
                }))
            },
        )
        .map_err(|err| err.to_string())
}

pub fn grammar_unit_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    Ok(json!({
        "id": row.get::<_, i64>(0)?,
        "title": row.get::<_, String>(1)?,
        "language": row.get::<_, String>(2)?,
        "level": row.get::<_, String>(3)?,
        "section": row.get::<_, String>(4)?,
        "tags": row.get::<_, Option<String>>(5)?,
        "theory": row.get::<_, String>(6)?,
        "examples": row.get::<_, String>(7)?,
        "status": row.get::<_, String>(8)?,
        "mastery_streak": row.get::<_, i64>(9)?,
        "last_score": row.get::<_, Option<f64>>(10)?,
        "last_practiced_at": row.get::<_, Option<String>>(11)?,
        "next_review_at": row.get::<_, Option<String>>(12)?,
        "exercise_count": row.get::<_, i64>(13)?,
        "linked_cards_count": row.get::<_, i64>(14)?,
        "created_at": row.get::<_, String>(15)?,
        "updated_at": row.get::<_, String>(16)?,
    }))
}

pub fn list_grammar_units(connection: &Connection) -> Result<Value, String> {
    let mut stmt = connection
        .prepare(
            r#"
            SELECT u.id, u.title, u.language, u.level, u.section, u.tags, u.theory, u.examples,
                   u.status, u.mastery_streak, u.last_score, u.last_practiced_at, u.next_review_at,
                   (SELECT COUNT(*) FROM grammar_exercises e WHERE e.unit_id = u.id),
                   (SELECT COUNT(*) FROM cards c WHERE u.tags IS NOT NULL AND u.tags != '' AND (
                      ',' || lower(REPLACE(c.tags, ' ', '')) || ','
                      LIKE '%,' || lower(REPLACE(u.tags, ' ', '')) || ',%'
                   )),
                   u.created_at, u.updated_at
            FROM grammar_units u
            ORDER BY
              CASE u.status WHEN 'in_progress' THEN 0 WHEN 'not_started' THEN 1 ELSE 2 END,
              COALESCE(datetime(u.next_review_at), datetime('2999-12-31')) ASC,
              u.section ASC, u.title ASC
            "#,
        )
        .map_err(|err| err.to_string())?;
    let rows = stmt
        .query_map([], grammar_unit_row)
        .map_err(|err| err.to_string())?;
    Ok(Value::Array(
        rows.collect::<Result<Vec<_>, _>>().map_err(|err| err.to_string())?,
    ))
}

pub fn create_grammar_unit(connection: &Connection, input: GrammarUnitInput) -> Result<Value, String> {
    connection
        .execute(
            r#"
            INSERT INTO grammar_units (title, language, level, section, tags, theory, examples)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![
                input.title.trim(), input.language.trim(), input.level.trim(),
                input.section.trim(), normalize_tags(input.tags),
                input.theory.trim(), input.examples.trim(),
            ],
        )
        .map_err(|err| err.to_string())?;
    Ok(json!({ "id": connection.last_insert_rowid() }))
}

pub fn update_grammar_unit(connection: &Connection, unit_id: i64, input: GrammarUnitInput) -> Result<(), String> {
    connection
        .execute(
            r#"
            UPDATE grammar_units
            SET title = ?1, language = ?2, level = ?3, section = ?4, tags = ?5,
                theory = ?6, examples = ?7, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?8
            "#,
            params![
                input.title.trim(), input.language.trim(), input.level.trim(),
                input.section.trim(), normalize_tags(input.tags),
                input.theory.trim(), input.examples.trim(), unit_id,
            ],
        )
        .map_err(|err| err.to_string())?;
    Ok(())
}

pub fn delete_grammar_unit(connection: &Connection, unit_id: i64) -> Result<(), String> {
    connection
        .execute("DELETE FROM grammar_units WHERE id = ?1", [unit_id])
        .map_err(|err| err.to_string())?;
    Ok(())
}

fn grammar_exercise_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    Ok(json!({
        "id": row.get::<_, i64>(0)?,
        "unit_id": row.get::<_, i64>(1)?,
        "type": row.get::<_, String>(2)?,
        "prompt": row.get::<_, String>(3)?,
        "payload": row.get::<_, String>(4)?,
        "answer": row.get::<_, String>(5)?,
        "explanation": row.get::<_, String>(6)?,
        "created_at": row.get::<_, String>(7)?,
    }))
}

pub fn list_grammar_exercises(connection: &Connection, unit_id: i64) -> Result<Value, String> {
    let mut stmt = connection
        .prepare(
            r#"
            SELECT id, unit_id, type, prompt, payload, answer, explanation, created_at
            FROM grammar_exercises
            WHERE unit_id = ?1
            ORDER BY type ASC, id ASC
            "#,
        )
        .map_err(|err| err.to_string())?;
    let rows = stmt
        .query_map([unit_id], grammar_exercise_row)
        .map_err(|err| err.to_string())?;
    Ok(Value::Array(
        rows.collect::<Result<Vec<_>, _>>().map_err(|err| err.to_string())?,
    ))
}

pub fn save_grammar_exercise(connection: &Connection, input: GrammarExerciseInput) -> Result<Value, String> {
    connection
        .execute(
            r#"
            INSERT INTO grammar_exercises (unit_id, type, prompt, payload, answer, explanation)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
            params![
                input.unit_id, input.exercise_type.trim(), input.prompt.trim(),
                input.payload.trim(), input.answer.trim(), input.explanation.trim(),
            ],
        )
        .map_err(|err| err.to_string())?;
    Ok(json!({ "id": connection.last_insert_rowid() }))
}

pub fn get_grammar_practice(connection: &Connection, unit_id: i64, limit: Option<i64>) -> Result<Value, String> {
    let max = limit.unwrap_or(8).clamp(3, 24);
    let mut stmt = connection
        .prepare(
            r#"
            SELECT id, unit_id, type, prompt, payload, answer, explanation, created_at
            FROM grammar_exercises
            WHERE unit_id = ?1
            ORDER BY RANDOM()
            LIMIT ?2
            "#,
        )
        .map_err(|err| err.to_string())?;
    let rows = stmt
        .query_map(params![unit_id, max], grammar_exercise_row)
        .map_err(|err| err.to_string())?;
    Ok(Value::Array(
        rows.collect::<Result<Vec<_>, _>>().map_err(|err| err.to_string())?,
    ))
}

pub fn complete_grammar_session(
    connection: &Connection,
    unit_id: i64,
    correct_count: i64,
    total_count: i64,
) -> Result<Value, String> {
    let total = total_count.max(1);
    let correct = correct_count.max(0).min(total);
    let score = (correct as f64 / total as f64) * 100.0;
    let current_streak: i64 = connection
        .query_row(
            "SELECT mastery_streak FROM grammar_units WHERE id = ?1",
            [unit_id],
            |row| row.get(0),
        )
        .map_err(|err| err.to_string())?;
    let next_streak = if score >= 80.0 {
        current_streak + 1
    } else if score < 60.0 {
        0
    } else {
        current_streak
    };
    let status = if next_streak >= 3 { "mastered" } else { "in_progress" };
    let next_review_at = if status == "mastered" {
        Some((Local::now() + Duration::days(14)).format("%Y-%m-%d %H:%M:%S").to_string())
    } else if score < 60.0 {
        Some(Local::now().format("%Y-%m-%d %H:%M:%S").to_string())
    } else {
        None
    };

    connection
        .execute(
            "INSERT INTO grammar_sessions (unit_id, score, correct_count, total_count) VALUES (?1, ?2, ?3, ?4)",
            params![unit_id, score, correct, total],
        )
        .map_err(|err| err.to_string())?;
    connection
        .execute(
            r#"
            UPDATE grammar_units
            SET status = ?1, mastery_streak = ?2, last_score = ?3,
                last_practiced_at = CURRENT_TIMESTAMP, next_review_at = ?4,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?5
            "#,
            params![status, next_streak, score, next_review_at, unit_id],
        )
        .map_err(|err| err.to_string())?;

    Ok(json!({
        "unit_id":       unit_id,
        "score":         score.round(),
        "correct_count": correct,
        "total_count":   total,
        "status":        status,
        "mastery_streak": next_streak,
        "next_review_at": next_review_at,
    }))
}

pub fn check_grammar_unit(connection: &Connection, title: &str) -> Result<Value, String> {
    let count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM grammar_units WHERE title = ?1",
            [title],
            |row| row.get(0),
        )
        .unwrap_or(0);
    Ok(json!({ "count": count }))
}

fn tag_patterns(tags: &str) -> Vec<String> {
    tags.split(',')
        .map(|tag| tag.trim().to_lowercase().replace(' ', ""))
        .filter(|tag| !tag.is_empty())
        .map(|tag| format!("%,{},%", tag))
        .collect()
}

pub fn get_grammar_linked_cards(connection: &Connection, unit_id: i64) -> Result<Value, String> {
    let tags: Option<String> = connection
        .query_row(
            "SELECT tags FROM grammar_units WHERE id = ?1",
            [unit_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|err| err.to_string())?
        .flatten();
    let Some(tags) = tags else {
        return Ok(Value::Array(Vec::new()));
    };
    let patterns = tag_patterns(&tags);
    if patterns.is_empty() {
        return Ok(Value::Array(Vec::new()));
    }
    let filter = patterns
        .iter()
        .map(|_| "(',' || lower(REPLACE(tags, ' ', '')) || ',') LIKE ?")
        .collect::<Vec<_>>()
        .join(" OR ");
    let mut stmt = connection
        .prepare(&format!(
            r#"
            SELECT id, deck_id, front, back, context, transcription, etymology, mnemonic,
                   notes, example_1, example_2, example_3, audio_path, image_path,
                   video_path, source, tags, status, due_at, last_reviewed_at,
                   interval_days, ease_factor, repetitions, lapses, created_at, updated_at
            FROM cards
            WHERE tags IS NOT NULL AND ({})
            ORDER BY updated_at DESC
            LIMIT 24
            "#,
            filter
        ))
        .map_err(|err| err.to_string())?;
    let rows = stmt
        .query_map(params_from_iter(patterns.iter()), |row| {
            Ok(json!({
                "id": row.get::<_, i64>(0)?,
                "deck_id": row.get::<_, i64>(1)?,
                "front": row.get::<_, String>(2)?,
                "back": row.get::<_, String>(3)?,
                "context": row.get::<_, Option<String>>(4)?,
                "transcription": row.get::<_, Option<String>>(5)?,
                "etymology": row.get::<_, Option<String>>(6)?,
                "mnemonic": row.get::<_, Option<String>>(7)?,
                "notes": row.get::<_, Option<String>>(8)?,
                "example_1": row.get::<_, Option<String>>(9)?,
                "example_2": row.get::<_, Option<String>>(10)?,
                "example_3": row.get::<_, Option<String>>(11)?,
                "audio_path": row.get::<_, Option<String>>(12)?,
                "image_path": row.get::<_, Option<String>>(13)?,
                "video_path": row.get::<_, Option<String>>(14)?,
                "source": row.get::<_, Option<String>>(15)?,
                "tags": row.get::<_, Option<String>>(16)?,
                "status": row.get::<_, String>(17)?,
                "due_at": row.get::<_, String>(18)?,
                "last_reviewed_at": row.get::<_, Option<String>>(19)?,
                "interval_days": row.get::<_, f64>(20)?,
                "ease_factor": row.get::<_, f64>(21)?,
                "repetitions": row.get::<_, i64>(22)?,
                "lapses": row.get::<_, i64>(23)?,
                "created_at": row.get::<_, String>(24)?,
                "updated_at": row.get::<_, String>(25)?,
            }))
        })
        .map_err(|err| err.to_string())?;
    Ok(Value::Array(
        rows.collect::<Result<Vec<_>, _>>().map_err(|err| err.to_string())?,
    ))
}

pub fn get_grammar_units_for_card(connection: &Connection, card_id: i64) -> Result<Value, String> {
    let tags: Option<String> = connection
        .query_row(
            "SELECT tags FROM cards WHERE id = ?1",
            [card_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|err| err.to_string())?
        .flatten();
    let Some(tags) = tags else {
        return Ok(Value::Array(Vec::new()));
    };
    let patterns = tag_patterns(&tags);
    if patterns.is_empty() {
        return Ok(Value::Array(Vec::new()));
    }
    let filter = patterns
        .iter()
        .map(|_| "(',' || lower(REPLACE(u.tags, ' ', '')) || ',') LIKE ?")
        .collect::<Vec<_>>()
        .join(" OR ");
    let mut stmt = connection
        .prepare(&format!(
            r#"
            SELECT u.id, u.title, u.language, u.level, u.section, u.tags, u.theory, u.examples,
                   u.status, u.mastery_streak, u.last_score, u.last_practiced_at, u.next_review_at,
                   (SELECT COUNT(*) FROM grammar_exercises e WHERE e.unit_id = u.id),
                   0,
                   u.created_at, u.updated_at
            FROM grammar_units u
            WHERE u.tags IS NOT NULL AND u.tags != '' AND ({})
            ORDER BY u.status ASC, u.title ASC
            LIMIT 6
            "#,
            filter
        ))
        .map_err(|err| err.to_string())?;
    let rows = stmt
        .query_map(params_from_iter(patterns.iter()), grammar_unit_row)
        .map_err(|err| err.to_string())?;
    Ok(Value::Array(
        rows.collect::<Result<Vec<_>, _>>().map_err(|err| err.to_string())?,
    ))
}
