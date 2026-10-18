# Dorlane

Dorlane is a local-first desktop application for legal teams. It combines case management, document storage, audit trails, local AI workflows, and knowledge-base search in a Tauri, React, TypeScript, Rust, and SQLite stack.

The product is designed for sensitive legal work: core data stays on the user's machine, runtime folders are isolated from source code, and privileged actions are routed through explicit application services.

## Product Scope

- Client, case, line, appeal, and document management.
- Local knowledge base with full-text and semantic search.
- Local AI assistance for legal research, drafting, risk scanning, and dossier workflows.
- Role-aware user administration and audit logging.
- Structured initialization pipeline for folders, databases, migrations, and security prerequisites.
- Tauri desktop shell with Rust-backed file system, SQLite, AI, and indexing commands.

## Architecture

```text
src/
  components/        Shared UI and application layout
  core/              Application shell composition
  i18n/              Runtime language configuration and locale files
  pages/             User-facing product screens
  routes/            Application routing
  services/          Domain, AI, database, filesystem, security, and RAG services
  store/             Zustand stores
  types/             Shared TypeScript entities

src-tauri/
  src/               Rust commands and native integrations
  Cargo.toml         Tauri/Rust dependency manifest

tests/               Lightweight regression checks
tools/               Project validation and release utilities
```

## Data Model

Dorlane creates a user-selected workspace and stores operational data there. Runtime folders and local databases are intentionally ignored by git.

```text
Dorlane/
  Clients/
  Knowledge_Base/
  Legal_Corpus/
  PII_Vault/
  .db_user/
  .db_clients/
  .db_kb/
```

Secrets, local databases, generated binaries, model files, build outputs, logs, and workspace data must not be committed.

## Development

Install dependencies:

```bash
pnpm install
```

Run the web development server:

```bash
pnpm dev
```

Run the Tauri application:

```bash
pnpm tauri dev
```

Build the frontend:

```bash
pnpm build
```

## Validation

Use the consolidated validation script before demos, releases, and investor walkthroughs:

```bash
pnpm ci:validate
```

Individual checks:

```bash
pnpm typecheck
pnpm test
pnpm lint
pnpm check:translations
pnpm build
```

Rust checks can be run from the Tauri package:

```bash
cd src-tauri
cargo check
```

## Security Notes

- `.env`, local databases, generated workspace folders, and model files are excluded from version control.
- SQLite and vector-search functionality are provided through the Rust/Tauri layer.
- PII and legal-corpus storage are separated from general workspace folders.
- Application logs are routed through the internal logger instead of ad hoc UI debug output.

## Investor Demo Readiness

Before presenting the project:

1. Run `pnpm ci:validate`.
2. Run `cargo check` in `src-tauri`.
3. Start a clean local workspace.
4. Verify onboarding, client creation, document upload, knowledge-base indexing, semantic search, and audit log views.
5. Confirm no secrets, databases, model binaries, or personal workspaces are staged in git.
