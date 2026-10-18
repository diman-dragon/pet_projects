use super::types::DocType;
use super::detector_helpers::{any, count_substr, is_arbitration};

fn sample(text: &str) -> &str {
    let end = text.char_indices().nth(3000).map(|(i,_)| i).unwrap_or(text.len());
    &text[..end]
}

/// Определяет тип юридического документа по структурным маркерам.
pub fn detect_doc_type(text: &str) -> DocType {
    let s = sample(text);

    // Арбитражный суд — проверяем ПЕРВЫМ
    if is_arbitration(s) && any(s, &["УСТАНОВИЛ", "ОПРЕДЕЛИЛ", "РЕШИЛ", "ПОСТАНОВИЛ"]) {
        return DocType::ArbitrationCourt;
    }
    if any(s, &["РЕШИЛ:", "ОПРЕДЕЛИЛ:", "ПОСТАНОВИЛ:"]) { return DocType::Court; }
    if any(s, &["УСТАНОВИЛ:", "МОТИВИРОВОЧНАЯ ЧАСТЬ", "РЕЗОЛЮТИВНАЯ ЧАСТЬ"]) {
        return DocType::Procedural;
    }
    if count_substr(text, "Статья ") >= 2 { return DocType::Codex; }
    if super::detector_helpers::count_contract_lines(text) >= 2 { return DocType::Contract; }
    DocType::Generic
}
