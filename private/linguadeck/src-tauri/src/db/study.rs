use rusqlite::{params, params_from_iter, Connection};
use serde_json::{json, Value};
use chrono::{Duration, Local, Datelike};
use super::cards::load_card;
use super::settings::load_settings;

pub fn get_study_session_cards(connection: &Connection, deck_id: i64) -> Result<Value, String> {
    let settings = load_settings(connection)?;
    let max = settings.max_reviews_per_session;

    // 1. Due cards
    let due_ids: Vec<i64> = {
        let mut stmt = connection
            .prepare(
                "SELECT id FROM cards \
                 WHERE deck_id = ? \
                   AND status != 'new' \
                   AND datetime(due_at) <= datetime('now', 'localtime') \
                 ORDER BY \
                   CASE status WHEN 'learning' THEN 0 WHEN 'review' THEN 1 ELSE 2 END, \
                   datetime(due_at) ASC, \
                   lapses DESC, \
                   COALESCE((SELECT AVG(rating) FROM review_logs rl WHERE rl.card_id = cards.id AND rl.rating_source = 'auto'), 4) ASC, \
                   COALESCE((SELECT AVG(front_time_ms) FROM review_logs rl WHERE rl.card_id = cards.id AND rl.rating_source = 'auto' AND front_time_ms > 0), 0) DESC, \
                   RANDOM() \
                 LIMIT ?",
            )
            .map_err(|e| e.to_string())?;
        
        let rows = stmt.query_map(params![deck_id, max], |row| row.get(0))
            .map_err(|e| e.to_string())?;
            
        let mut ids = Vec::new();
        for id in rows {
            ids.push(id.map_err(|e| e.to_string())?);
        }
        ids
    };

    let due_count = due_ids.len() as i64;
    let mut cards: Vec<Value> = Vec::new();
    for id in &due_ids {
        cards.push(load_card(connection, *id)?);
    }

    // 2. New cards
    let new_slots = (max - due_count).max(0);
    if new_slots > 0 {
        let new_ids: Vec<i64> = {
            let mut stmt = connection
                .prepare(
                    "SELECT id FROM cards \
                     WHERE deck_id = ? \
                       AND status = 'new' \
                       AND id NOT IN ( \
                           SELECT DISTINCT card_id FROM review_logs \
                           WHERE rating_source = 'auto' AND date(reviewed_at, 'localtime') = date('now', 'localtime') \
                       ) \
                     ORDER BY RANDOM() \
                     LIMIT ?",
                )
                .map_err(|e| e.to_string())?;
            
            let rows = stmt.query_map(params![deck_id, new_slots], |row| row.get(0))
                .map_err(|e| e.to_string())?;
                
            let mut ids = Vec::new();
            for id in rows {
                ids.push(id.map_err(|e| e.to_string())?);
            }
            ids
        };
        for id in &new_ids {
            cards.push(load_card(connection, *id)?);
        }
    }

    // 3. Fallback new cards
    let remaining_slots = (max - cards.len() as i64).max(0);
    if remaining_slots > 0 {
        let existing_ids: Vec<i64> = cards.iter().map(|c| c["id"].as_i64().unwrap_or(0)).collect();
        let placeholders = existing_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = if existing_ids.is_empty() {
            format!("SELECT id FROM cards WHERE deck_id = ? AND status = 'new' ORDER BY RANDOM() LIMIT ?")
        } else {
            format!("SELECT id FROM cards WHERE deck_id = ? AND status = 'new' AND id NOT IN ({}) ORDER BY RANDOM() LIMIT ?", placeholders)
        };
        let mut stmt = connection.prepare(&sql).map_err(|e| e.to_string())?;
        let mut params_vec: Vec<&dyn rusqlite::ToSql> = vec![&deck_id];
        for id in &existing_ids { params_vec.push(id); }
        params_vec.push(&remaining_slots);
        
        let ids: Vec<i64> = stmt.query_map(params_from_iter(params_vec.iter()), |row| row.get(0))
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
            
        for id in ids {
            cards.push(load_card(connection, id)?);
        }
    }

    // 4. Learning queue boost
    let boost_slots = (max - cards.len() as i64).max(0);
    if boost_slots > 0 {
        let existing_ids: Vec<i64> = cards.iter().map(|c| c["id"].as_i64().unwrap_or(0)).collect();
        let placeholders = existing_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = if existing_ids.is_empty() {
            format!("SELECT id FROM cards WHERE deck_id = ? AND status = 'learning' ORDER BY datetime(due_at) ASC, lapses DESC LIMIT ?")
        } else {
            format!("SELECT id FROM cards WHERE deck_id = ? AND status = 'learning' AND id NOT IN ({}) ORDER BY datetime(due_at) ASC, lapses DESC LIMIT ?", placeholders)
        };
        let mut stmt = connection.prepare(&sql).map_err(|e| e.to_string())?;
        let mut params_vec: Vec<&dyn rusqlite::ToSql> = vec![&deck_id];
        for id in &existing_ids { params_vec.push(id); }
        params_vec.push(&boost_slots);
        
        let ids: Vec<i64> = stmt.query_map(params_from_iter(params_vec.iter()), |row| row.get(0))
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
            
        for id in ids {
            cards.push(load_card(connection, id)?);
        }
    }

    Ok(Value::Array(cards))
}

pub fn get_study_forecast(connection: &Connection, days: i64) -> Result<Value, String> {
    let today = Local::now().date_naive();
    let mut result = Vec::new();
    for i in 0..days {
        let date = today + Duration::days(i);
        let date_str = date.format("%Y-%m-%d").to_string();
        let due_count: i64 = connection.query_row(
            "SELECT COUNT(*) FROM cards WHERE status IN ('learning', 'review') AND date(due_at) = date(?1)",
            [&date_str],
            |row| row.get(0),
        ).unwrap_or(0);
        let new_count: i64 = if i == 0 {
            connection.query_row("SELECT COUNT(*) FROM cards WHERE status = 'new'", [], |row| row.get(0)).unwrap_or(0)
        } else { 0 };
        result.push(json!({ "date": date_str, "due": due_count, "new": new_count, "total": due_count + new_count }));
    }
    Ok(Value::Array(result))
}

pub fn get_study_cards_for_day(connection: &Connection, date: &str) -> Result<Value, String> {
    let mut stmt = connection.prepare(
        "SELECT id FROM cards WHERE status IN ('learning', 'review') AND date(due_at) = date(?1) ORDER BY datetime(due_at) ASC",
    ).map_err(|e| e.to_string())?;
    
    let ids: Vec<i64> = stmt.query_map(params![date], |row| row.get::<_, i64>(0))
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
        
    let mut cards = Vec::new();
    for id in ids { cards.push(load_card(connection, id)?); }
    Ok(Value::Array(cards))
}

pub fn get_views_for_day(connection: &Connection, date: &str) -> Result<Value, String> {
    let mut stmt = connection.prepare(
        "SELECT vl.id, vl.card_id, vl.viewed_at, vl.time_spent_ms, c.front, c.back, c.deck_id, d.name as deck_name \
         FROM view_logs vl JOIN cards c ON vl.card_id = c.id JOIN decks d ON c.deck_id = d.id \
         WHERE date(vl.viewed_at, 'localtime') = date(?1) ORDER BY vl.viewed_at DESC",
    ).map_err(|e| e.to_string())?;
    
    let rows: Vec<Value> = stmt.query_map(params![date], |row| {
        Ok(json!({
            "id": row.get::<_, i64>(0)?, "card_id": row.get::<_, i64>(1)?,
            "viewed_at": row.get::<_, String>(2)?, "time_spent_ms": row.get::<_, i64>(3)?,
            "front": row.get::<_, String>(4)?, "back": row.get::<_, String>(5)?,
            "deck_id": row.get::<_, i64>(6)?, "deck_name": row.get::<_, String>(7)?,
        }))
    }).map_err(|e| e.to_string())?.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?;
    
    Ok(Value::Array(rows))
}

pub fn get_activity_history(connection: &Connection, days: i64) -> Result<Value, String> {
    let today = Local::now().date_naive();
    let mut rows = Vec::new();
    for i in 0..days {
        let date = today - Duration::days(i);
        let date_str = date.format("%Y-%m-%d").to_string();
        let reviewed: i64 = connection.query_row(
            "SELECT COUNT(*) FROM review_logs WHERE rating_source = 'auto' AND date(reviewed_at, 'localtime') = date(?1)",
            [&date_str], |row| row.get(0),
        ).unwrap_or(0);
        let viewed: i64 = connection.query_row(
            "SELECT COUNT(*) FROM view_logs WHERE date(viewed_at, 'localtime') = date(?1)",
            [&date_str], |row| row.get(0),
        ).unwrap_or(0);
        let time_ms: i64 = connection.query_row(
            "SELECT COALESCE(SUM(time_spent_ms), 0) FROM view_logs WHERE date(viewed_at, 'localtime') = date(?1)",
            [&date_str], |row| row.get(0),
        ).unwrap_or(0);
        rows.push(json!({ "date": date_str, "count": reviewed + viewed, "reviewed": reviewed, "viewed": viewed, "time_ms": time_ms }));
    }
    Ok(Value::Array(rows))
}

pub fn get_day_stats(connection: &Connection, date: &str) -> Result<Value, String> {
    let views: i64 = connection.query_row("SELECT COUNT(*) FROM view_logs WHERE date(viewed_at, 'localtime') = date(?1)", [&date], |row| row.get(0)).unwrap_or(0);
    let unique: i64 = connection.query_row("SELECT COUNT(DISTINCT card_id) FROM view_logs WHERE date(viewed_at, 'localtime') = date(?1)", [&date], |row| row.get(0)).unwrap_or(0);
    let flips: i64 = connection.query_row("SELECT COUNT(*) FROM view_logs WHERE view_type = 'view_flipped' AND date(viewed_at, 'localtime') = date(?1)", [&date], |row| row.get(0)).unwrap_or(0);
    let time_ms: i64 = connection.query_row("SELECT COALESCE(SUM(time_spent_ms), 0) FROM view_logs WHERE date(viewed_at, 'localtime') = date(?1)", [&date], |row| row.get(0)).unwrap_or(0);
    let reviewed: i64 = connection.query_row("SELECT COUNT(*) FROM review_logs WHERE rating_source = 'auto' AND date(reviewed_at, 'localtime') = date(?1)", [&date], |row| row.get(0)).unwrap_or(0);
    let correct: i64 = connection.query_row("SELECT COUNT(*) FROM review_logs WHERE rating_source = 'auto' AND date(reviewed_at, 'localtime') = date(?1) AND rating >= 3", [&date], |row| row.get(0)).unwrap_or(0);
    let accuracy = if reviewed > 0 { (correct as f64 / reviewed as f64 * 100.0).round() } else { 0.0 };
    Ok(json!({ "views": views, "unique_cards": unique, "flips": flips, "reviewed": reviewed, "time_ms": time_ms, "accuracy": accuracy, "correct": correct, "total": views }))
}

pub fn get_streak(connection: &Connection) -> Result<Value, String> {
    let today = Local::now().date_naive();
    let mut current_streak = 0;
    for offset in 0..365 {
        let day = today - Duration::days(offset);
        let day_str = day.format("%Y-%m-%d").to_string();
        let activity: i64 = connection.query_row("SELECT COUNT(*) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1", [&day_str], |row| row.get(0)).unwrap_or(0);
        if activity > 0 { current_streak += 1; } else if offset == 0 { continue; } else { break; }
    }
    let reviewed_today: i64 = connection.query_row("SELECT COUNT(*) FROM review_logs WHERE rating_source = 'auto' AND date(reviewed_at, 'localtime') = date('now', 'localtime')", [], |row| row.get(0)).unwrap_or(0);
    let time_today_ms: i64 = connection.query_row("SELECT COALESCE(SUM(time_spent_ms), 0) FROM view_logs WHERE date(viewed_at, 'localtime') = date('now', 'localtime')", [], |row| row.get(0)).unwrap_or(0);
    let mut week_activity = Vec::new();
    for offset in (0..7).rev() {
        let day = today - Duration::days(offset);
        let day_str = day.format("%Y-%m-%d").to_string();
        let count: i64 = connection.query_row("SELECT COUNT(DISTINCT card_id) FROM view_logs WHERE date(viewed_at, 'localtime') = ?1", [&day_str], |row| row.get(0)).unwrap_or(0);
        week_activity.push(json!({ "date": day_str, "count": count, "weekday": day.weekday().num_days_from_monday() }));
    }
    let message = if current_streak == 0 { "Начни серию сегодня".to_string() } else { format!("{} дней подряд", current_streak) };
    Ok(json!({ "current": current_streak, "session_count": reviewed_today, "reviewed_today": reviewed_today, "time_today_ms": time_today_ms, "week_activity": week_activity, "is_active": current_streak > 0, "message": message }))
}
