pub fn any(text: &str, pats: &[&str]) -> bool {
    pats.iter().any(|p| text.contains(p))
}

pub fn count_substr(text: &str, pat: &str) -> usize {
    let mut n = 0; let mut s = text;
    while let Some(p) = s.find(pat) { n += 1; s = &s[p + pat.len()..]; }
    n
}

/// Арбитражный суд: дело «АNN-XXXXXX/YY» + «арбитражный суд».
pub fn is_arbitration(text: &str) -> bool {
    let has_court = text.contains("Арбитражный суд") || text.contains("арбитражный суд");
    if !has_court { return false; }
    // Ищем паттерн А{2-3цифры дела}-число/год в байтах (UTF-8 'А' = 0xD090)
    let b = text.as_bytes();
    for i in 0..b.len().saturating_sub(4) {
        // Кириллическая А: D0 90
        if b[i] == 0xD0 && i+1 < b.len() && b[i+1] == 0x90 && i+2 < b.len() && b[i+2].is_ascii_digit() {
            return true;
        }
        // ASCII A + digit (транслитерированный вариант)
        if b[i] == b'A' && i+1 < b.len() && b[i+1].is_ascii_digit() {
            return true;
        }
    }
    false
}

pub fn count_contract_lines(text: &str) -> usize {
    text.lines().filter(|l| {
        let t = l.trim_start();
        if t.len() < 3 { return false; }
        let mut it = t.chars();
        let c1 = it.next().unwrap_or_default();
        if !c1.is_ascii_digit() { return false; }
        let rest: String = std::iter::once(c1).chain(it.take(3)).collect();
        rest.find('.').map(|dp| {
            if dp == 0 || dp > 2 { return false; }
            rest[dp+1..].trim_start().chars().next()
                .map(|c| c.is_uppercase() || c == '«' || c == '"').unwrap_or(false)
        }).unwrap_or(false)
    }).count()
}
