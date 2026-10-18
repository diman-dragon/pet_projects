pub struct BehaviorMetrics {
    pub front_time_ms: i64,
    pub total_time_ms: i64,
    pub flips_count: i64,
    pub backtracks_count: i64,
    pub manual_wrong: bool,
}

pub fn infer_rating(m: &BehaviorMetrics) -> i64 {
    // 1. Manual override (Backspace / 'X' click)
    if m.manual_wrong {
        return 1;
    }

    // 2. Outlier detection (Coffee break filter)
    // If total time > 60s, we ignore it for grading and return Good (3) 
    // to not ruin the card interval or Ease factor.
    if m.total_time_ms > 60_000 {
        return 3;
    }

    // 3. Progressive time-based grading (based on Time to Reveal - front_time_ms)
    let t_front = m.front_time_ms as f64 / 1000.0;
    let flips = m.flips_count;

    // High friction (many flips) usually indicates difficulty
    if flips >= 3 {
        return 1;
    }

    // Rating based on "Progressive Scale 3-8-15-20"
    if t_front < 3.0 {
        return 4; // Easy: Instant recognition
    }
    if t_front < 8.0 {
        return 3; // Good: Fast enough
    }
    if t_front < 15.0 {
        return 2; // Hard: Significant effort
    }
    if t_front < 20.0 {
        return 2; // Hard boundary
    }
    
    1 // Again: Too slow or forgot
}
