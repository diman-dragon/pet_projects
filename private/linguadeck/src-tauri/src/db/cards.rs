use rusqlite::{params, params_from_iter, Connection};
use serde_json::{json, Value};
use chrono::{Duration, Local};
use super::models::{SettingsInput, StudyBehaviorInput, CardInput};
use super::connection::now_string;
use super::utils::{parse_learning_steps, normalize_tags};
use super::settings::load_settings;

pub fn load_card(connection: &Connection, card_id: i64) -> Result<Value, String> {
    connection
        .query_row(
            r#"
            SELECT id, deck_id, front, back, context, transcription, etymology, mnemonic,
                   notes, example_1, example_2, example_3, audio_path, image_path,
                   video_path, source, tags, status, due_at, last_reviewed_at,
                   interval_days, ease_factor, repetitions, lapses, created_at, updated_at
            FROM cards
            WHERE id = ?1
            "#,
            [card_id],
            |row| {
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
            },
        )
        .map_err(|err| err.to_string())
}

pub fn create_card(connection: &Connection, deck_id: i64, input: CardInput) -> Result<(), String> {
    connection
        .execute(
            r#"
            INSERT INTO cards (
              deck_id, front, back, context, transcription, etymology, mnemonic, notes,
              example_1, example_2, example_3, audio_path, image_path, video_path, source, tags
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
            "#,
            params![
                deck_id, input.front.trim(), input.back.trim(), input.context, input.transcription,
                input.etymology, input.mnemonic, input.notes, input.example_1, input.example_2,
                input.example_3, input.audio_path, input.image_path, input.video_path,
                input.source, normalize_tags(input.tags)
            ],
        )
        .map_err(|err| err.to_string())?;
    update_deck_card_count(connection, deck_id)?;
    Ok(())
}

pub fn create_card_batch(
    connection: &Connection,
    deck_id: i64,
    inputs: Vec<CardInput>,
) -> Result<i64, String> {
    if inputs.is_empty() {
        return Ok(0);
    }
    connection
        .execute("BEGIN", [])
        .map_err(|err| err.to_string())?;

    let mut inserted: i64 = 0;
    for input in inputs {
        let front = input.front.trim().to_string();
        let back = input.back.trim().to_string();
        if front.is_empty() || back.is_empty() {
            continue;
        }
        connection
            .execute(
                r#"
                INSERT INTO cards (
                  deck_id, front, back, context, transcription, etymology, mnemonic, notes,
                  example_1, example_2, example_3, audio_path, image_path, video_path, source, tags
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
                "#,
                params![
                    deck_id, front, back,
                    input.context, input.transcription, input.etymology, input.mnemonic,
                    input.notes, input.example_1, input.example_2, input.example_3,
                    input.audio_path, input.image_path, input.video_path,
                    input.source, normalize_tags(input.tags)
                ],
            )
            .map_err(|err| {
                let _ = connection.execute("ROLLBACK", []);
                err.to_string()
            })?;
        inserted += 1;
    }

    connection
        .execute("COMMIT", [])
        .map_err(|err| err.to_string())?;
    update_deck_card_count(connection, deck_id)?;
    Ok(inserted)
}

pub fn update_card(connection: &Connection, card_id: i64, input: CardInput) -> Result<(), String> {
    connection
        .execute(
            r#"
            UPDATE cards
            SET front = ?1, back = ?2, context = ?3, transcription = ?4, etymology = ?5,
                mnemonic = ?6, notes = ?7, example_1 = ?8, example_2 = ?9, example_3 = ?10,
                audio_path = ?11, image_path = ?12, video_path = ?13, source = ?14, tags = ?15,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?16
            "#,
            params![
                input.front.trim(), input.back.trim(), input.context, input.transcription,
                input.etymology, input.mnemonic, input.notes, input.example_1, input.example_2,
                input.example_3, input.audio_path, input.image_path, input.video_path,
                input.source, normalize_tags(input.tags), card_id
            ],
        )
        .map_err(|err| err.to_string())?;
    Ok(())
}

pub fn delete_card(connection: &Connection, card_id: i64, deck_id: i64) -> Result<(), String> {
    connection
        .execute("DELETE FROM cards WHERE id = ?1", [card_id])
        .map_err(|err| err.to_string())?;
    update_deck_card_count(connection, deck_id)?;
    Ok(())
}

pub fn delete_cards_many(connection: &Connection, card_ids: Vec<i64>, deck_id: i64) -> Result<(), String> {
    if card_ids.is_empty() {
        return Ok(());
    }
    let placeholders = std::iter::repeat("?")
        .take(card_ids.len())
        .collect::<Vec<_>>()
        .join(", ");
    let query = format!("DELETE FROM cards WHERE id IN ({})", placeholders);
    connection
        .execute(&query, params_from_iter(card_ids.iter()))
        .map_err(|err| err.to_string())?;
    update_deck_card_count(connection, deck_id)?;
    Ok(())
}

pub fn list_cards(connection: &Connection, deck_id: i64) -> Result<Value, String> {
    let mut stmt = connection
        .prepare(
            "SELECT id FROM cards WHERE deck_id = ?1 ORDER BY updated_at DESC, id DESC",
        )
        .map_err(|err| err.to_string())?;
    let ids = stmt
        .query_map([deck_id], |row| row.get::<_, i64>(0))
        .map_err(|err| err.to_string())?;
    let mut cards = Vec::new();
    for id in ids {
        cards.push(load_card(connection, id.map_err(|err| err.to_string())?)?);
    }
    Ok(Value::Array(cards))
}

pub fn get_card_stats(connection: &Connection, card_id: i64) -> Result<Value, String> {
    let (views_total, flips_total, avg_view_ms, total_time_ms): (i64, i64, f64, i64) = connection
        .query_row(
            r#"
            SELECT
              COALESCE(SUM(CASE WHEN view_type = 'view' THEN 1 ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN view_type = 'view_flipped' THEN 1 ELSE 0 END), 0),
              COALESCE(AVG(CASE WHEN view_type = 'view' AND time_spent_ms > 0 THEN CAST(time_spent_ms AS REAL) END), 0),
              COALESCE(SUM(time_spent_ms), 0)
            FROM view_logs WHERE card_id = ?1
            "#,
            [card_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .map_err(|err| err.to_string())?;

    let mut stmt = connection
        .prepare(
            r#"
            SELECT rating, interval_before, interval_after, ease_before, ease_after, reviewed_at,
                   rating_source, front_time_ms, total_time_ms, flips_count, backtracks_count
            FROM review_logs
            WHERE card_id = ?1 AND rating_source = 'auto'
            ORDER BY reviewed_at DESC
            LIMIT 10
            "#,
        )
        .map_err(|err| err.to_string())?;

    let reviews = stmt
        .query_map([card_id], |row| {
            Ok(json!({
                "rating": row.get::<_, i64>(0)?,
                "interval_before": row.get::<_, f64>(1)?,
                "interval_after": row.get::<_, f64>(2)?,
                "ease_before": row.get::<_, f64>(3)?,
                "ease_after": row.get::<_, f64>(4)?,
                "reviewed_at": row.get::<_, String>(5)?,
                "rating_source": row.get::<_, String>(6)?,
                "front_time_ms": row.get::<_, i64>(7)?,
                "total_time_ms": row.get::<_, i64>(8)?,
                "flips_count": row.get::<_, i64>(9)?,
                "backtracks_count": row.get::<_, i64>(10)?,
            }))
        })
        .map_err(|err| err.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| err.to_string())?;

    let (reviews_total, avg_rating, best_rating, worst_rating): (i64, f64, i64, i64) = connection
        .query_row(
            r#"
            SELECT
              COUNT(*),
              COALESCE(AVG(CAST(rating AS REAL)), 0),
              COALESCE(MAX(rating), 0),
              COALESCE(MIN(rating), 0)
            FROM review_logs WHERE card_id = ?1 AND rating_source = 'auto'
            "#,
            [card_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .map_err(|err| err.to_string())?;

    Ok(json!({
        "views_total": views_total,
        "flips_total": flips_total,
        "avg_view_ms": avg_view_ms.round() as i64,
        "total_time_ms": total_time_ms,
        "reviews_total": reviews_total,
        "avg_rating": avg_rating,
        "best_rating": best_rating,
        "worst_rating": worst_rating,
        "recent_reviews": reviews,
    }))
}

pub fn update_deck_card_count(connection: &Connection, deck_id: i64) -> Result<(), String> {
    connection
        .execute(
            r#"
            UPDATE decks
            SET cards_count = (SELECT COUNT(*) FROM cards WHERE deck_id = ?1),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?1
            "#,
            [deck_id],
        )
        .map_err(|err| err.to_string())?;
    Ok(())
}

pub fn due_order_case(algorithm: &str) -> &'static str {
    if algorithm == "cram" {
        "CASE status WHEN 'new' THEN 0 WHEN 'learning' THEN 1 WHEN 'review' THEN 2 ELSE 3 END"
    } else {
        "CASE status WHEN 'learning' THEN 0 WHEN 'review' THEN 1 WHEN 'new' THEN 2 ELSE 3 END"
    }
}

pub fn card_numbers(card: &Value) -> (f64, f64, i64, i64) {
    (
        card["ease_factor"].as_f64().unwrap_or(2.5),
        card["interval_days"].as_f64().unwrap_or(0.0),
        card["repetitions"].as_i64().unwrap_or(0),
        card["lapses"].as_i64().unwrap_or(0),
    )
}

pub fn schedule_next(card: &Value, rating: i64, settings: &SettingsInput) -> Value {
    if settings.study_algorithm == "fsrs" {
        return schedule_fsrs(card, rating, settings);
    }
    let (ease_before, interval_before, repetitions_before, lapses_before) = card_numbers(card);
    let steps = parse_learning_steps(&settings.learning_steps);
    let first_step = steps.first().copied().unwrap_or(1);
    let second_step = steps.get(1).copied().unwrap_or((first_step + 9).max(10));
    let mut due_date = Local::now();
    let (ease_after, interval_after, repetitions_after, lapses_after, status) =
        if settings.study_algorithm == "cram" {
            let repetitions_after = if rating >= 3 { repetitions_before + 1 } else { 0 };
            if rating == 1 {
                (ease_before, 0.0, repetitions_after, lapses_before + 1, "new")
            } else {
                due_date += Duration::minutes(first_step);
                (ease_before, 0.0, repetitions_after, lapses_before, "learning")
            }
        } else if rating == 1 {
            due_date += Duration::minutes(first_step);
            ((ease_before - 0.2).max(1.3), 0.0, 0, lapses_before + 1, "learning")
        } else {
            let repetitions_after = repetitions_before + 1;
            let interval_after = if repetitions_after == 1 {
                if rating == 4 {
                    let interval = settings.easy_interval.max(2) as f64;
                    due_date += Duration::days(interval as i64);
                    return json!({
                        "intervalAfter": interval,
                        "easeAfter": (ease_before + 0.1).max(1.3),
                        "repetitionsAfter": repetitions_after,
                        "lapsesAfter": lapses_before,
                        "dueAt": due_date.format("%Y-%m-%d %H:%M:%S").to_string(),
                        "status": "review",
                    });
                }
                due_date += Duration::minutes(second_step);
                0.0
            } else if repetitions_after == 2 {
                let interval = if rating == 4 {
                    settings.easy_interval as f64
                } else {
                    settings.graduating_interval as f64
                };
                due_date += Duration::days(interval as i64);
                interval
            } else {
                let bonus = if rating == 4 { 1.2 } else if rating == 3 { 1.0 } else { 0.75 };
                let interval = (interval_before * ease_before * bonus).round().max(1.0);
                due_date += Duration::days(interval as i64);
                interval
            };
            let diff = 4 - rating;
            let sm2_adjustment = 0.1 - (diff as f64) * (0.08 + (diff as f64) * 0.02);
            let status = if repetitions_after == 1 { "learning" } else { "review" };
            (
                (ease_before + sm2_adjustment).max(1.3),
                interval_after,
                repetitions_after,
                lapses_before,
                status,
            )
        };
    json!({
        "intervalAfter": interval_after,
        "easeAfter": ease_after,
        "repetitionsAfter": repetitions_after,
        "lapsesAfter": lapses_after,
        "dueAt": due_date.format("%Y-%m-%d %H:%M:%S").to_string(),
        "status": status,
    })
}

pub fn schedule_fsrs(card: &Value, rating: i64, settings: &SettingsInput) -> Value {
    let (ease_before, interval_before, repetitions_before, lapses_before) = card_numbers(card);
    let difficulty_before = (5.0 + (2.5 - ease_before) * 2.0).clamp(1.0, 10.0);
    let stability_before = if interval_before > 0.0 {
        interval_before / 0.9_f64.ln() * 0.5_f64.ln()
    } else {
        0.1
    };
    let steps = parse_learning_steps(&settings.learning_steps);
    let first_step = steps.first().copied().unwrap_or(1);
    let second_step = steps.get(1).copied().unwrap_or(10);
    let w_d = [0.4, 0.6, 0.8, 1.0];
    let w_s = [0.0, 0.5, 1.0, 2.0];
    let mut due_date = Local::now();
    let (stability_after, difficulty_after, interval_after, repetitions_after, lapses_after, status) =
        if rating == 1 {
            let retrievability = (-interval_before / stability_before.max(0.1)).exp();
            (
                stability_before * (1.0 - retrievability) * 0.1,
                difficulty_before + w_d[0] * (5.0 - difficulty_before),
                0.0,
                0,
                lapses_before + 1,
                "learning",
            )
        } else {
            let reps = repetitions_before + 1;
            let difficulty = difficulty_before - w_d[(rating - 1) as usize] * (difficulty_before - 1.0);
            if reps == 1 {
                if rating == 4 {
                    due_date += Duration::days(settings.easy_interval);
                    (settings.easy_interval as f64, difficulty, settings.easy_interval as f64, reps, lapses_before, "review")
                } else if rating == 3 {
                    due_date += Duration::minutes(second_step);
                    (1.0, difficulty, 0.0, reps, lapses_before, "learning")
                } else {
                    due_date += Duration::minutes(first_step);
                    (0.5, difficulty, 0.0, reps, lapses_before, "learning")
                }
            } else if reps == 2 && rating < 4 {
                due_date += Duration::days(settings.graduating_interval);
                (
                    settings.graduating_interval as f64,
                    difficulty,
                    settings.graduating_interval as f64,
                    reps,
                    lapses_before,
                    "review",
                )
            } else {
                let retrievability = (-interval_before / stability_before.max(0.1)).exp();
                let bonus = 1.0
                    + w_s[(rating - 1) as usize] * (1.0 - retrievability) * stability_before.powf(-0.5);
                let stability = stability_before * bonus;
                let interval = (stability * 0.9_f64.ln() / 0.5_f64.ln()).round().max(1.0);
                due_date += Duration::days(interval as i64);
                (stability, difficulty, interval, reps, lapses_before, "review")
            }
        };
    let ease_after = (2.5 - (difficulty_after - 5.0) / 2.0).clamp(1.3, 2.5);
    json!({
        "intervalAfter": interval_after,
        "easeAfter": ease_after,
        "repetitionsAfter": repetitions_after,
        "lapsesAfter": lapses_after,
        "dueAt": due_date.format("%Y-%m-%d %H:%M:%S").to_string(),
        "status": status,
        "stabilityAfter": stability_after,
    })
}

pub fn apply_card_review(
    connection: &Connection,
    card_id: i64,
    rating: i64,
    metrics: Option<&StudyBehaviorInput>,
) -> Result<Value, String> {
    let card = load_card(connection, card_id)?;
    let settings = load_settings(connection)?;
    let next = schedule_next(&card, rating.clamp(1, 4), &settings);
    connection
        .execute(
            r#"
            UPDATE cards
            SET status = ?1, due_at = ?2, last_reviewed_at = CURRENT_TIMESTAMP,
                interval_days = ?3, ease_factor = ?4, repetitions = ?5, lapses = ?6,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?7
            "#,
            params![
                next["status"].as_str().unwrap_or("learning"),
                next["dueAt"].as_str().unwrap_or(&now_string()),
                next["intervalAfter"].as_f64().unwrap_or(0.0),
                next["easeAfter"].as_f64().unwrap_or(2.5),
                next["repetitionsAfter"].as_i64().unwrap_or(0),
                next["lapsesAfter"].as_i64().unwrap_or(0),
                card_id
            ],
        )
        .map_err(|err| err.to_string())?;

    let front_time_ms = metrics.map(|m| m.front_time_ms.max(0)).unwrap_or(0);
    let total_time_ms = metrics.map(|m| m.total_time_ms.max(0)).unwrap_or(0);
    let flips_count = metrics.map(|m| m.flips_count.max(0)).unwrap_or(0);
    let backtracks_count = metrics.map(|m| m.backtracks_count.max(0)).unwrap_or(0);

    connection
        .execute(
            r#"
            INSERT INTO review_logs (
                card_id, deck_id, rating, rating_source,
                front_time_ms, total_time_ms, flips_count, backtracks_count,
                interval_before, interval_after, ease_before, ease_after, reviewed_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, datetime('now'))
            "#,
            params![
                card_id,
                card["deck_id"].as_i64().unwrap_or_default(),
                rating,
                "auto",
                front_time_ms,
                total_time_ms,
                flips_count,
                backtracks_count,
                card["interval_days"].as_f64().unwrap_or(0.0),
                next["intervalAfter"].as_f64().unwrap_or(0.0),
                card["ease_factor"].as_f64().unwrap_or(2.5),
                next["easeAfter"].as_f64().unwrap_or(2.5),
            ],
        )
        .map_err(|err| err.to_string())?;

    Ok(json!({
        "rating": rating,
        "ratingSource": "auto",
        "schedule": next,
    }))
}
