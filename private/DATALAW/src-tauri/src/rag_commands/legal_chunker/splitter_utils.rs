use regex::Regex;

/// Разбивает текст на предложения по знакам «. ? !» перед заглавной.
pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut last = 0;
    let b = text.as_bytes();
    for i in 1..b.len().saturating_sub(1) {
        if matches!(b[i], b'.' | b'?' | b'!') && matches!(b[i+1], b' ' | b'\n') {
            result.push(&text[last..=i]);
            last = i + 2;
            while last < b.len() && b[last] == b' ' { last += 1; }
        }
    }
    if last < text.len() { result.push(&text[last..]); }
    result
}

pub fn char_count(s: &str) -> usize { s.chars().count() }

/// Берём последние `n` символов строки как перекрытие.
pub fn take_tail(s: &str, n: usize) -> String {
    let total = char_count(s);
    if n >= total { return s.to_string(); }
    s.chars().skip(total - n).collect()
}

pub fn extract_article_ref(text: &str) -> Option<String> {
    let patterns = [
        r"(?i)(?:ст(?:атья)?|art(?:icle)?|§)\s*\.?\s*(\d+(?:\.\d+)?(?:\s*[а-яёa-z])?(?:\s*п\.\s*\d+)?)",
        r"(?i)Rule\s+(\d+)",
        r"(?i)(?:Правило|п(?:ункт)?)\s*(\d+(?:\.\d+)?)",
    ];

    for pattern in patterns {
        if let Ok(re) = Regex::new(pattern) {
            if let Some(cap) = re.captures(text) {
                return Some(cap[0].chars().take(80).collect());
            }
        }
    }
    None
}

pub fn split_into_children(text: &str, child_size: usize) -> Vec<(usize, usize, String)> {
    let mut children = Vec::new();
    let sentences = split_sentences(text);
    
    let mut current = String::new();
    let mut current_start = 0;
    let mut offset = 0;

    for sentence in sentences {
        let s_len = char_count(sentence);
        if !current.is_empty() && char_count(&current) + s_len > child_size {
            children.push((current_start, offset, current.trim().to_string()));
            current.clear();
            current_start = offset;
        }
        current.push_str(sentence);
        offset += s_len;
    }

    if !current.trim().is_empty() {
        children.push((current_start, offset, current.trim().to_string()));
    }

    children
}
