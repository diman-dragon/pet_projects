use super::types::MAX_HEADER_LEN;

/// Обрезает строку до MAX_HEADER_LEN символов.
pub fn trim_header(s: &str) -> String {
    let s = s.trim().replace('\n', " ");
    let count = s.chars().count();
    if count <= MAX_HEADER_LEN {
        s
    } else {
        let end = s.char_indices()
            .nth(MAX_HEADER_LEN - 1)
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        format!("{}…", &s[..end])
    }
}

/// Строит строку вида `[DocTitle | Глава 29 | Статья 450] `.
/// None-части пропускаются.
pub fn build_breadcrumb(parts: &[Option<String>]) -> String {
    let valid: Vec<&str> = parts
        .iter()
        .filter_map(|p| p.as_deref())
        .filter(|s| !s.is_empty())
        .collect();

    if valid.is_empty() {
        String::new()
    } else {
        format!("[{}] ", valid.join(" | "))
    }
}

/// Строит полный текст чанка = breadcrumb + header + "\n" + body.
/// Если body пуст — опускаем перевод строки.
pub fn compose_chunk(breadcrumb: &str, header: &str, body: &str) -> String {
    let header_part = if header.is_empty() {
        String::new()
    } else {
        format!("{}\n", header)
    };
    format!("{}{}{}", breadcrumb, header_part, body.trim())
}
