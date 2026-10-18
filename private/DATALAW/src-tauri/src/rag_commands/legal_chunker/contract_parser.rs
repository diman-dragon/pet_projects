/// Парсит заголовок раздела договора «1.», «1.1.», «12.3.»
/// Возвращает (номер, глубина) или None.
pub fn parse_section_header(line: &str) -> Option<(String, usize)> {
    let t = line.trim();
    if t.len() < 3 { return None; }
    let mut num_str = String::new();
    let mut last_byte = 0;
    let mut dot_count = 0usize;
    for (i, c) in t.char_indices() {
        if c.is_ascii_digit() { num_str.push(c); last_byte = i + 1; }
        else if c == '.' { dot_count += 1; num_str.push('.'); last_byte = i + 1; }
        else { break; }
    }
    if num_str.is_empty() || dot_count == 0 { return None; }
    let rest = t[last_byte..].trim_start();
    if rest.is_empty() { return None; }
    let first = rest.chars().next()?;
    if !first.is_uppercase() && first != '«' && first != '"' && !first.is_ascii_digit() {
        return None;
    }
    let depth = num_str.trim_end_matches('.').split('.').count();
    Some((num_str.trim_end_matches('.').to_string(), depth))
}
