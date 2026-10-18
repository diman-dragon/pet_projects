use super::splitter_utils::{split_sentences, char_count, take_tail};
use super::types::{RawChunk, OVERLAP_CHARS};

/// Разбивает длинный текст на части ≤ max_chars (по абзацам, затем по предложениям).
pub fn split_long(text: &str, max_chars: usize, min_chars: usize) -> Vec<String> {
    if char_count(text) <= max_chars { return vec![text.trim().to_string()]; }

    // Попытка 1: по двойному переводу строки
    let paras: Vec<&str> = text.split("\n\n").map(str::trim)
        .filter(|s| char_count(s) >= min_chars).collect();
    if paras.len() > 1 {
        let mut res = Vec::new(); let mut cur = String::new();
        for para in &paras {
            if !cur.is_empty() && char_count(&cur) + 2 + char_count(para) > max_chars {
                res.push(std::mem::take(&mut cur));
                cur = para.to_string();
            } else {
                if !cur.is_empty() { cur.push_str("\n\n"); }
                cur.push_str(para);
            }
        }
        if char_count(&cur) >= min_chars { res.push(cur); }
        if !res.is_empty() { return res; }
    }

    // Попытка 2: по предложениям с перекрытием
    let sents = split_sentences(text);
    let mut res = Vec::new(); let mut cur = String::new();
    for s in sents {
        if !cur.is_empty() && char_count(&cur) + char_count(s) > max_chars {
            if char_count(&cur) >= min_chars { res.push(std::mem::take(&mut cur)); }
            cur = take_tail(&cur, OVERLAP_CHARS);
        }
        cur.push_str(s);
    }
    if char_count(&cur) >= min_chars { res.push(cur); }
    res
}

/// Преобразует вектор строк в RawChunk с корректными offsets.
pub fn finalize(texts: Vec<String>, base_offset: usize, min_chars: usize) -> Vec<RawChunk> {
    let mut offset = base_offset;
    texts.into_iter().filter(|t| char_count(t) >= min_chars).map(|text| {
        let len = char_count(&text);
        let chunk = RawChunk { text, char_start: offset, char_end: offset + len };
        offset += len + 2;
        chunk
    }).collect()
}
