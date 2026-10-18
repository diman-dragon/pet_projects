use super::breadcrumb::{build_breadcrumb, compose_chunk, trim_header};
use super::codex_context::{classify_line, CodexContext};
use super::splitter::{split_long, finalize};
use super::types::{RawChunk, DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS};

pub fn chunk_codex(text: &str, doc_title: &str, max_chars: usize, min_chars: usize) -> Vec<RawChunk> {
    let max = max_chars.max(DEFAULT_MAX_CHARS);
    let min = min_chars.max(DEFAULT_MIN_CHARS);
    let mut result = Vec::new();
    let mut ctx = CodexContext::default();
    let mut header = String::new();
    let mut body = String::new();
    let mut char_pos: usize = 0;
    let mut section_start: usize = 0;

    let flush = |ctx: &CodexContext, hdr: &str, body: &str, start: usize, res: &mut Vec<RawChunk>| {
        if body.trim().is_empty() { return; }
        let crumb = build_breadcrumb(&[
            Some(trim_header(doc_title)),
            ctx.part.clone(), ctx.section.clone(),
            ctx.chapter.clone(), ctx.article.clone(),
        ]);
        let full = compose_chunk(&crumb, hdr, body);
        if full.chars().count() <= max {
            res.extend(finalize(vec![full], start, min));
        } else {
            for part in split_long(body, max.saturating_sub(crumb.len() + hdr.len() + 1), min) {
                res.extend(finalize(vec![compose_chunk(&crumb, hdr, &part)], start, min));
            }
        }
    };

    for line in text.lines() {
        let line_len = line.chars().count() + 1;
        if let Some((level, hdr_text)) = classify_line(line) {
            flush(&ctx, &header, &body, section_start, &mut result);
            body.clear(); section_start = char_pos;
            header = trim_header(&hdr_text);
            ctx.update(level, &hdr_text);
        } else {
            body.push_str(line); body.push('\n');
        }
        char_pos += line_len;
    }
    flush(&ctx, &header, &body, section_start, &mut result);
    result
}
