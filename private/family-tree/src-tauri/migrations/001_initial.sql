CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS id_sequences (
    name TEXT PRIMARY KEY NOT NULL,
    next_value INTEGER NOT NULL CHECK (next_value > 0)
);

INSERT OR IGNORE INTO id_sequences (name, next_value) VALUES ('person', 1);

CREATE TABLE IF NOT EXISTS persons (
    id TEXT PRIMARY KEY NOT NULL,
    is_private INTEGER NOT NULL DEFAULT 0 CHECK (is_private IN (0, 1)),
    sex TEXT,
    gender_identity TEXT,
    biological_sex TEXT,
    is_living INTEGER CHECK (is_living IN (0, 1)),
    is_deceased INTEGER CHECK (is_deceased IN (0, 1)),
    deceased_status TEXT,
    nationality TEXT,
    ethnicity TEXT,
    religion TEXT,
    occupation TEXT,
    education TEXT,
    description TEXT,
    biography TEXT,
    research_status TEXT,
    photo_url TEXT,
    photo_zoom REAL NOT NULL DEFAULT 1,
    photo_offset_x REAL NOT NULL DEFAULT 0,
    photo_offset_y REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS date_values (
    id TEXT PRIMARY KEY NOT NULL,
    original_text TEXT NOT NULL,
    calendar TEXT NOT NULL DEFAULT 'gregorian',
    date_kind TEXT NOT NULL,
    precision TEXT NOT NULL,
    year INTEGER,
    month INTEGER CHECK (month IS NULL OR month BETWEEN 1 AND 12),
    day INTEGER CHECK (day IS NULL OR day BETWEEN 1 AND 31),
    lower_bound TEXT,
    upper_bound TEXT,
    lower_bound_kind TEXT,
    upper_bound_kind TEXT,
    qualifier TEXT,
    sort_date TEXT,
    sort_precision TEXT,
    phrase TEXT,
    normalized_text TEXT,
    gedcom_text TEXT
);

CREATE TABLE IF NOT EXISTS places (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    normalized_name TEXT,
    place_type TEXT,
    latitude REAL,
    longitude REAL,
    country_code TEXT,
    valid_from_date_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    valid_to_date_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    historical_name TEXT,
    current_name TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS place_names (
    id TEXT PRIMARY KEY NOT NULL,
    place_id TEXT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    language TEXT,
    script TEXT,
    name_type TEXT,
    valid_from_date_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    valid_to_date_id TEXT REFERENCES date_values(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS person_names (
    id TEXT PRIMARY KEY NOT NULL,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    name_type TEXT NOT NULL DEFAULT 'birth',
    full_text TEXT NOT NULL,
    given_name TEXT,
    patronymic TEXT,
    surname TEXT,
    maiden_name TEXT,
    prefix TEXT,
    suffix TEXT,
    nickname TEXT,
    name_order INTEGER,
    language TEXT,
    script TEXT,
    transliteration TEXT,
    is_preferred INTEGER NOT NULL DEFAULT 0 CHECK (is_preferred IN (0, 1)),
    valid_from_date_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    valid_to_date_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY NOT NULL,
    event_type TEXT NOT NULL,
    date_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    place_id TEXT REFERENCES places(id) ON DELETE SET NULL,
    title TEXT,
    description TEXT,
    cause TEXT,
    status TEXT,
    is_primary INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0, 1)),
    is_private INTEGER NOT NULL DEFAULT 0 CHECK (is_private IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS event_participants (
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'principal',
    sequence INTEGER,
    notes TEXT,
    PRIMARY KEY (event_id, person_id, role)
);

CREATE TABLE IF NOT EXISTS person_facts (
    id TEXT PRIMARY KEY NOT NULL,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    fact_type TEXT NOT NULL,
    value_text TEXT,
    value_number REAL,
    value_boolean INTEGER CHECK (value_boolean IS NULL OR value_boolean IN (0, 1)),
    date_from_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    date_to_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    place_id TEXT REFERENCES places(id) ON DELETE SET NULL,
    certainty TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY NOT NULL,
    person_a_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    person_b_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    lineage_type TEXT,
    date_from_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    date_to_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    certainty TEXT,
    is_preferred INTEGER NOT NULL DEFAULT 0 CHECK (is_preferred IN (0, 1)),
    notes TEXT,
    CHECK (person_a_id <> person_b_id)
);

CREATE TABLE IF NOT EXISTS unions (
    id TEXT PRIMARY KEY NOT NULL,
    union_type TEXT NOT NULL,
    date_from_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    date_to_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    place_id TEXT REFERENCES places(id) ON DELETE SET NULL,
    status TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS union_members (
    union_id TEXT NOT NULL REFERENCES unions(id) ON DELETE CASCADE,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    role TEXT,
    sequence INTEGER,
    PRIMARY KEY (union_id, person_id)
);

CREATE TABLE IF NOT EXISTS union_children (
    union_id TEXT NOT NULL REFERENCES unions(id) ON DELETE CASCADE,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    child_status TEXT,
    sequence INTEGER,
    PRIMARY KEY (union_id, person_id)
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    publisher TEXT,
    publication_date_id TEXT REFERENCES date_values(id) ON DELETE SET NULL,
    repository TEXT,
    archive_id TEXT,
    call_number TEXT,
    url TEXT,
    language TEXT,
    source_type TEXT,
    description TEXT,
    copyright TEXT,
    access_date_id TEXT REFERENCES date_values(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS source_citations (
    id TEXT PRIMARY KEY NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    page TEXT,
    volume TEXT,
    chapter TEXT,
    entry TEXT,
    quotation TEXT,
    citation_text TEXT,
    quality TEXT,
    confidence TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS event_sources (
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    citation_id TEXT NOT NULL REFERENCES source_citations(id) ON DELETE CASCADE,
    PRIMARY KEY (event_id, citation_id)
);

CREATE TABLE IF NOT EXISTS person_sources (
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    citation_id TEXT NOT NULL REFERENCES source_citations(id) ON DELETE CASCADE,
    PRIMARY KEY (person_id, citation_id)
);

CREATE TABLE IF NOT EXISTS person_fact_sources (
    fact_id TEXT NOT NULL REFERENCES person_facts(id) ON DELETE CASCADE,
    citation_id TEXT NOT NULL REFERENCES source_citations(id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, citation_id)
);

CREATE TABLE IF NOT EXISTS relationship_sources (
    relationship_id TEXT NOT NULL REFERENCES relationships(id) ON DELETE CASCADE,
    citation_id TEXT NOT NULL REFERENCES source_citations(id) ON DELETE CASCADE,
    PRIMARY KEY (relationship_id, citation_id)
);

CREATE TABLE IF NOT EXISTS media (
    id TEXT PRIMARY KEY NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    mime_type TEXT,
    checksum TEXT,
    width INTEGER,
    height INTEGER,
    caption TEXT,
    description TEXT,
    media_type TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS media_links (
    media_id TEXT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
    person_id TEXT REFERENCES persons(id) ON DELETE CASCADE,
    event_id TEXT REFERENCES events(id) ON DELETE CASCADE,
    source_id TEXT REFERENCES sources(id) ON DELETE CASCADE,
    relationship_id TEXT REFERENCES relationships(id) ON DELETE CASCADE,
    sort_order INTEGER,
    CHECK (person_id IS NOT NULL OR event_id IS NOT NULL OR source_id IS NOT NULL OR relationship_id IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY NOT NULL,
    text TEXT NOT NULL,
    author TEXT,
    is_private INTEGER NOT NULL DEFAULT 0 CHECK (is_private IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS note_links (
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    subject_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    PRIMARY KEY (note_id, subject_type, subject_id)
);

CREATE TABLE IF NOT EXISTS tags (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL UNIQUE,
    color TEXT
);

CREATE TABLE IF NOT EXISTS tag_links (
    tag_id TEXT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    subject_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    PRIMARY KEY (tag_id, subject_type, subject_id)
);

CREATE TABLE IF NOT EXISTS research_assertions (
    id TEXT PRIMARY KEY NOT NULL,
    subject_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    assertion_type TEXT NOT NULL,
    value TEXT,
    certainty TEXT,
    reasoning TEXT,
    status TEXT NOT NULL DEFAULT 'hypothesis',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS assertion_sources (
    assertion_id TEXT NOT NULL REFERENCES research_assertions(id) ON DELETE CASCADE,
    citation_id TEXT NOT NULL REFERENCES source_citations(id) ON DELETE CASCADE,
    PRIMARY KEY (assertion_id, citation_id)
);

CREATE TABLE IF NOT EXISTS canvas_cards (
    id TEXT PRIMARY KEY NOT NULL,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    x REAL NOT NULL,
    y REAL NOT NULL,
    width REAL NOT NULL DEFAULT 250,
    height REAL NOT NULL DEFAULT 350,
    generation INTEGER NOT NULL DEFAULT 0,
    is_visible INTEGER NOT NULL DEFAULT 1 CHECK (is_visible IN (0, 1)),
    is_pinned INTEGER NOT NULL DEFAULT 0 CHECK (is_pinned IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS canvas_views (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    pan_x REAL NOT NULL DEFAULT 0,
    pan_y REAL NOT NULL DEFAULT 0,
    zoom REAL NOT NULL DEFAULT 1,
    is_default INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_person_names_person_id ON person_names(person_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_event_participants_person_id ON event_participants(person_id);
CREATE INDEX IF NOT EXISTS idx_person_facts_person_id ON person_facts(person_id);
CREATE INDEX IF NOT EXISTS idx_relationships_person_a ON relationships(person_a_id);
CREATE INDEX IF NOT EXISTS idx_relationships_person_b ON relationships(person_b_id);
CREATE INDEX IF NOT EXISTS idx_union_members_person_id ON union_members(person_id);
CREATE INDEX IF NOT EXISTS idx_union_children_person_id ON union_children(person_id);
CREATE INDEX IF NOT EXISTS idx_canvas_cards_person_id ON canvas_cards(person_id);

CREATE TABLE IF NOT EXISTS canvas_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1)
);

INSERT OR IGNORE INTO canvas_settings (id) VALUES (1);
CREATE INDEX IF NOT EXISTS idx_note_links_subject ON note_links(subject_type, subject_id);
CREATE INDEX IF NOT EXISTS idx_tag_links_subject ON tag_links(subject_type, subject_id);

INSERT OR IGNORE INTO schema_migrations (version) VALUES (1);
