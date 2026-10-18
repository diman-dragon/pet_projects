use super::rating::{BehaviorMetrics, infer_rating};
use super::models::StudyBehaviorInput;

pub fn normalize_tags(tags: Option<String>) -> Option<String> {
    tags.map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub fn parse_learning_steps(raw: &str) -> Vec<i64> {
    raw.split(',')
        .filter_map(|part| part.trim().parse::<i64>().ok())
        .filter(|step| *step > 0)
        .collect()
}

pub fn infer_behavior_rating(metrics: &StudyBehaviorInput) -> i64 {
    let m = BehaviorMetrics {
        front_time_ms: metrics.front_time_ms,
        total_time_ms: metrics.total_time_ms,
        flips_count: metrics.flips_count,
        backtracks_count: metrics.backtracks_count,
        manual_wrong: metrics.manual_wrong,
    };
    infer_rating(&m)
}
