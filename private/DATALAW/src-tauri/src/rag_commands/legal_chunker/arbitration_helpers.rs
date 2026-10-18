use regex::Regex;

pub const OPERATIVE_MARKERS: &[&str] = &[
    "УСТАНОВИЛ:", "ОПРЕДЕЛИЛ:", "РЕШИЛ:", "ПОСТАНОВИЛ:",
    "РАССМОТРЕВ:", "ИЗУЧИВ:", "МОТИВИРОВОЧНАЯ ЧАСТЬ", "РЕЗОЛЮТИВНАЯ ЧАСТЬ",
];

pub const PARTY_PREFIXES: &[&str] = &[
    "Заявителю", "Должнику", "Ответчику", "Истцу", "Кредитору",
    "Конкурсному управляющему", "Финансовому управляющему",
    "Временному управляющему", "Арбитражному управляющему", "Союзу",
];

pub fn extract_case_number(text: &str) -> Option<String> {
    let re = Regex::new(r"[АA]\d{2,3}-\d+/\d{2,4}").ok()?;
    re.find(text).map(|m| m.as_str().to_string())
}

pub fn extract_court_name(text: &str) -> String {
    text.lines().take(10)
        .find(|l| l.contains("суд") || l.contains("Суд"))
        .map(|l| l.trim().chars().take(80).collect())
        .unwrap_or_else(|| "Арбитражный суд".to_string())
}

pub fn is_operative_marker(line: &str) -> Option<String> {
    let t = line.trim();
    OPERATIVE_MARKERS.iter()
        .find(|&&m| t.starts_with(m.trim_end_matches('\n')))
        .map(|m| m.trim_end_matches([':', '\n']).to_string())
}

pub fn is_party_instruction(line: &str) -> Option<String> {
    let t = line.trim();
    PARTY_PREFIXES.iter()
        .find(|&&p| t.starts_with(p) && (t.contains(" – ") || t.contains(" — ") || t.ends_with(" –")))
        .map(|p| p.to_string())
}
