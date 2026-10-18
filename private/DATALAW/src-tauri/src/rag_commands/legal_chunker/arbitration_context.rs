use super::breadcrumb::{build_breadcrumb, trim_header};

/// Контекст арбитражного акта — суд, номер дела, breadcrumb-фабрика.
pub struct ArbContext {
    pub title: String,
    pub case_num: String,
}

impl ArbContext {
    pub fn new(doc_title: &str, court: &str, case_num: String) -> Self {
        Self {
            title: if doc_title.is_empty() { court.to_string() } else { doc_title.to_string() },
            case_num,
        }
    }

    pub fn crumb(&self, label: &str, party: Option<&str>) -> String {
        let label_str = match party {
            Some(p) => format!("{} | {}", label, p),
            None    => label.to_string(),
        };
        build_breadcrumb(&[
            Some(trim_header(&self.title)),
            if self.case_num.is_empty() { None } else { Some(self.case_num.clone()) },
            Some(label_str),
        ])
    }
}
