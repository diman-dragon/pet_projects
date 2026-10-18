use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use chrono::{Datelike, Duration, Local};

pub fn create_deck(
    connection: &Connection,
    course_id: i64,
    name: &str,
    description: &str,
    icon: &str,
    color: &str,
    language: Option<String>,
) -> Result<(), String> {
    let _ = connection.execute(
        "ALTER TABLE decks ADD COLUMN language TEXT NOT NULL DEFAULT ''",
        [],
    );
    connection
        .execute(
            "INSERT INTO decks (course_id, name, description, icon, color, language, cards_count) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0)",
            params![course_id, name.trim(), description.trim(), icon, color, language.unwrap_or_default()],
        )
        .map_err(|err| err.to_string())?;
    Ok(())
}

pub fn update_deck(
    connection: &Connection,
    deck_id: i64,
    name: &str,
    description: &str,
    icon: &str,
    color: &str,
    language: Option<String>,
) -> Result<(), String> {
    let _ = connection.execute(
        "ALTER TABLE decks ADD COLUMN language TEXT NOT NULL DEFAULT ''",
        [],
    );
    connection
        .execute(
            "UPDATE decks SET name = ?1, description = ?2, icon = ?3, color = ?4, language = ?5, updated_at = CURRENT_TIMESTAMP WHERE id = ?6",
            params![name.trim(), description.trim(), icon, color, language.unwrap_or_default(), deck_id],
        )
        .map_err(|err| err.to_string())?;
    Ok(())
}

pub fn delete_deck(connection: &Connection, deck_id: i64) -> Result<(), String> {
    connection
        .execute("DELETE FROM decks WHERE id = ?1", [deck_id])
        .map_err(|err| err.to_string())?;
    Ok(())
}

pub fn list_decks(connection: &Connection) -> Result<Value, String> {
    let _ = connection.execute(
        "ALTER TABLE decks ADD COLUMN language TEXT NOT NULL DEFAULT ''",
        [],
    );
    let mut stmt = connection
        .prepare(
            r#"
            SELECT d.id, d.course_id, d.name, d.description, d.icon, d.color, d.cards_count,
                   d.created_at, d.updated_at,
                   COALESCE(SUM(CASE WHEN c.status != 'new' AND datetime(c.due_at) <= datetime('now', 'localtime') THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN c.status = 'new' THEN 1 ELSE 0 END), 0),
                   COALESCE(d.language, '')
            FROM decks d
            LEFT JOIN cards c ON c.deck_id = d.id
            GROUP BY d.id
            ORDER BY d.updated_at DESC, d.id DESC
            "#,
        )
        .map_err(|err| err.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            Ok(json!({
                "id": row.get::<_, i64>(0)?,
                "course_id": row.get::<_, i64>(1)?,
                "name": row.get::<_, String>(2)?,
                "description": row.get::<_, String>(3)?,
                "icon": row.get::<_, String>(4)?,
                "color": row.get::<_, String>(5)?,
                "cards_count": row.get::<_, i64>(6)?,
                "created_at": row.get::<_, String>(7)?,
                "updated_at": row.get::<_, String>(8)?,
                "due_count": row.get::<_, i64>(9)?,
                "new_count": row.get::<_, i64>(10)?,
                "language": row.get::<_, String>(11)?,
            }))
        })
        .map_err(|err| err.to_string())?;
    let decks = rows.collect::<Result<Vec<_>, _>>().map_err(|err| err.to_string())?;
    Ok(Value::Array(decks))
}

pub fn get_dashboard_stats(connection: &Connection) -> Result<Value, String> {
    let today_str = Local::now().date_naive().format("%Y-%m-%d").to_string();

    let totals = connection
        .query_row(
            r#"
            SELECT
              (SELECT COUNT(*) FROM decks),
              (SELECT COUNT(*) FROM cards),
              (SELECT COUNT(*) FROM cards WHERE status = 'new'),
              (SELECT COUNT(*) FROM cards WHERE status = 'learning'),
              (SELECT COUNT(*) FROM cards WHERE status = 'review'),
              (SELECT COUNT(*) FROM cards WHERE status != 'new' AND due_at IS NOT NULL AND due_at <= strftime('%Y-%m-%dT%H:%M:%S', 'now')),
              (SELECT COALESCE(COUNT(*), 0) FROM review_logs WHERE rating_source = 'auto' AND date(reviewed_at, 'localtime') = ?1),
              (SELECT COALESCE(COUNT(*), 0) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1 AND view_type = 'view'),
              (SELECT COALESCE(COUNT(*), 0) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1 AND view_type = 'view_flipped'),
              (SELECT COUNT(*) FROM review_logs WHERE rating_source = 'auto'),
              (SELECT COUNT(DISTINCT card_id) FROM view_logs WHERE view_type = 'view'),
              (SELECT COUNT(DISTINCT card_id) FROM view_logs WHERE view_type = 'view_flipped'),
              (SELECT COUNT(*) FROM cards WHERE status = 'review' AND repetitions >= 3 AND interval_days >= 7),
              (SELECT COALESCE(AVG(ease_factor), 2.5) FROM cards WHERE repetitions > 0),
              (SELECT COALESCE(AVG(CASE WHEN rating >= 3 THEN 100.0 ELSE 0.0 END), 0) FROM review_logs WHERE rating_source = 'auto' AND date(reviewed_at, 'localtime') >= date('now', 'localtime', '-30 days')),
              (SELECT CASE WHEN COUNT(*) = 0 THEN 0 ELSE ROUND(100.0 * COUNT(CASE WHEN status != 'new' THEN 1 END) / COUNT(*), 1) END FROM cards),
              (SELECT COALESCE(COUNT(DISTINCT card_id), 0) FROM review_logs WHERE rating_source = 'auto' AND date(reviewed_at, 'localtime') = ?1 AND card_id IN (SELECT id FROM cards WHERE repetitions = 1)),
              (SELECT COALESCE(COUNT(*), 0) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1 AND view_type = 'view'),
              (SELECT COALESCE(COUNT(*), 0) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1 AND view_type = 'view_flipped'),
              (SELECT COALESCE(COUNT(*), 0) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1),
              (SELECT COALESCE(COUNT(DISTINCT card_id), 0) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1),
              (SELECT COALESCE(COUNT(*), 0) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1 AND view_type = 'view_flipped'),
              (SELECT COALESCE(SUM(time_spent_ms), 0) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1),
              (SELECT COALESCE(SUM(total_time_ms), 0) FROM review_logs WHERE rating_source = 'auto' AND date(reviewed_at, 'localtime') = ?1)
            "#,
            params![today_str],
            |row| {
                let total_cards: i64 = row.get(1)?;
                let new_cards: i64 = row.get(2)?;
                let learning_cards: i64 = row.get(3)?;
                let review_cards: i64 = row.get(4)?;
                let mastered_cards: i64 = row.get(12)?;
                let retention: f64 = row.get(14)?;
                let t = if total_cards > 0 { total_cards as f64 } else { 1.0 };
                let review_only = (review_cards - mastered_cards).max(0);
                // cols 19-22: total views today, unique cards today, flips today, time_ms today
                let shown_today: i64 = row.get(19)?;
                let unique_today: i64 = row.get(20)?;
                let flips_today: i64 = row.get(21)?;
                let view_time_today: i64 = row.get(22)?;
                // FIX: col 23 = review_logs total_time_ms today (ранее не учитывалось)
                let review_time_today: i64 = row.get(23)?;
                let time_today: i64 = view_time_today + review_time_today;
                Ok(json!({
                    "totalDecks": row.get::<_, i64>(0)?,
                    "totalCards": total_cards,
                    "newCards": new_cards,
                    "learningCards": learning_cards,
                    "reviewCards": review_cards,
                    "dueToday": row.get::<_, i64>(5)?,
                    "reviewedToday": row.get::<_, i64>(6)?,
                    "viewedToday": row.get::<_, i64>(7)?,
                    "viewedFlippedToday": row.get::<_, i64>(8)?,
                    "studiedAllTime": row.get::<_, i64>(9)?,
                    "viewedAllTime": row.get::<_, i64>(10)?,
                    "viewedFlippedAllTime": row.get::<_, i64>(11)?,
                    "masteredCards": mastered_cards,
                    "averageEase": row.get::<_, f64>(13)?,
                    "retentionRate": retention,
                    "retentionRounded": retention.round() as i64,
                    "completionRate": row.get::<_, f64>(15)?,
                    "newLearnedToday": row.get::<_, i64>(16)?,
                    "shownToday": shown_today,
                    "shownFlippedToday": flips_today,
                    "uniqueCardsToday": unique_today,
                    "timeTodayMs": time_today,
                    "newCardsPct": ((new_cards as f64 / t) * 100.0).round() as i64,
                    "learningCardsPct": ((learning_cards as f64 / t) * 100.0).round() as i64,
                    "reviewOnlyPct": ((review_only as f64 / t) * 100.0).round() as i64,
                    "masteredCardsPct": ((mastered_cards as f64 / t) * 100.0).round() as i64,
                }))
            },
        )
        .map_err(|err| err.to_string())?;

    let today = Local::now().date_naive();
    let mut streak: i64 = 0;
    for offset in 0..365 {
        let day = today - Duration::days(offset);
        let key = day.format("%Y-%m-%d").to_string();
        let exists: Option<i64> = connection
            .query_row(
                "SELECT 1 FROM view_logs WHERE date(viewed_at, 'localtime') = ?1 LIMIT 1",
                [&key],
                |row| row.get(0),
            )
            .optional()
            .map_err(|err| err.to_string())?;
        if exists.is_some() {
            streak += 1;
        } else if offset == 0 {
            // No activity today yet — don't break, just skip today
            continue;
        } else {
            break;
        }
    }

    let mut week_days: Vec<Value> = Vec::new();
    for offset in (0..7i64).rev() {
        let day = today - Duration::days(offset);
        let day_str = day.format("%Y-%m-%d").to_string();
        let weekday = day.weekday().num_days_from_monday() as i64;
        let label = match weekday {
            0 => "Пн", 1 => "Вт", 2 => "Ср", 3 => "Чт", 4 => "Пт", 5 => "Сб", _ => "Вс",
        };
        let views_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1",
                [&day_str],
                |row| row.get(0),
            )
            .unwrap_or(0);
        let unique_views_count: i64 = connection
            .query_row(
                "SELECT COUNT(DISTINCT card_id) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1",
                [&day_str],
                |row| row.get(0),
            )
            .unwrap_or(0);
        let flips_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM view_logs WHERE view_type = 'view_flipped' AND date(viewed_at, 'localtime') = ?1",
                [&day_str],
                |row| row.get(0),
            )
            .unwrap_or(0);
        let view_time_ms: i64 = connection
            .query_row(
                "SELECT COALESCE(SUM(time_spent_ms), 0) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1",
                [&day_str],
                |row| row.get(0),
            )
            .unwrap_or(0);
        let review_time_ms: i64 = connection
            .query_row(
                "SELECT COALESCE(SUM(total_time_ms), 0) FROM review_logs WHERE rating_source = 'auto' AND date(reviewed_at, 'localtime') = ?1",
                [&day_str],
                |row| row.get(0),
            )
            .unwrap_or(0);
        let total_cards = views_count;
        week_days.push(json!({
            "date": day_str,
            "label": label,
            "weekday": weekday,
            "count": total_cards,
            "uniqueCount": unique_views_count,
            "flipsCount": flips_count,
            "timeMs": view_time_ms + review_time_ms,
            "isToday": offset == 0,
            "hasActivity": total_cards > 0,
        }));
    }

    let mut output = totals;
    output["streakDays"] = json!(streak);
    output["weekActivity"] = json!(week_days);
    Ok(output)
}
