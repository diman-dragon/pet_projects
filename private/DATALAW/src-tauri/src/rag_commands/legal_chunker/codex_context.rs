use super::breadcrumb::trim_header;

/// Иерархический контекст НПА: запоминает заголовки родительских уровней.
#[derive(Clone, Default)]
pub struct CodexContext {
    pub part:    Option<String>,
    pub section: Option<String>,
    pub chapter: Option<String>,
    pub article: Option<String>,
}

pub fn classify_line(line: &str) -> Option<(&'static str, String)> {
    let t = line.trim();
    let cases: &[(&str, &str)] = &[
        ("Часть ",    "part"),    ("ЧАСТЬ ",    "part"),
        ("Раздел ",   "section"), ("РАЗДЕЛ ",   "section"),
        ("Глава ",    "chapter"), ("ГЛАВА ",    "chapter"),
        ("Статья ",   "article"), ("СТАТЬЯ ",   "article"),
        ("§ ",        "article"),
    ];
    for &(prefix, level) in cases {
        if t.starts_with(prefix) {
            return Some((level, t.to_string()));
        }
    }
    None
}

impl CodexContext {
    /// Обновляет контекст при переходе на новый уровень иерархии.
    pub fn update(&mut self, level: &str, header: &str) {
        let h = Some(trim_header(header));
        match level {
            "part"    => { self.part = h; self.section = None; self.chapter = None; self.article = None; }
            "section" => { self.section = h; self.chapter = None; self.article = None; }
            "chapter" => { self.chapter = h; self.article = None; }
            "article" => { self.article = h; }
            _ => {}
        }
    }
}
