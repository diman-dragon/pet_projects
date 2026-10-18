# Dorlane — Патч исправлений v0.9.5 → v0.9.6

## Что исправлено

### 🔴 БАГ 1 (КРИТИЧЕСКИЙ): КАД Арбитр → 404 Not Found
**Файл:** `src-tauri/src/services/client_dossier.rs`

- Исправлен endpoint: `/Result/Search` → `/Kad/SearchInstances`
- Исправлено тело запроса: ИНН теперь в поле `Sides[].Name` (не `Sides[].Inn`)
- Убрано несуществующее поле `WithDocuments`
- Добавлено обязательное поле `WithVKSInstances: false`
- Добавлен хедер `x-date-format: iso`
- Исправлен тот же endpoint в ОГРН-retry блоке

### 🔴 БАГ 2 (КРИТИЧЕСКИЙ): Тихий hash-fallback в эмбеддингах
**Файл:** `src/services/rag/embedding/local.ts`

- Добавлено явное предупреждение в console.warn при переходе в hash-режим
- Добавлен DOM-event `embedding:degraded` для UI-уведомлений
- Сброс флага деградации при успешном получении реального эмбеддинга

### 🔴 БАГ 3 (КРИТИЧЕСКИЙ): Rust struct теряет options чата
**Файл:** `src-tauri/src/commands/ai_commands.rs`

- Добавлен struct `OllamaOptions` с полями num_ctx, num_predict, temperature, top_p
- Добавлено поле `options: Option<OllamaOptions>` в `ChatCompletionRequest`
- Теперь параметры контекста корректно передаются в Ollama

### 🟠 БАГ 5 (ВЫСОКИЙ): Хардкод моделей в config.ts
**Файл:** `src/services/ai/config.ts`

- Убраны `AI_MODELS.widget = 'gemma2:2b'` и `AI_MODELS.analyze = 'gemma2:2b'`
- Добавлена функция `getActiveChatModel()` — берёт модель из ollamaStore
- Добавлена константа `DEFAULT_CHAT_MODEL = 'gemma3:4b'` как fallback
- `AI_MODELS.rag` оставлен — embedding-модель фиксирована

### 🔵 V2 (ВИЗУАЛЬНЫЙ): Нет панели embedding-модели в Settings
**Файл:** `src/pages/settings/components/OllamaSettingsSection.tsx`

- Добавлена панель "Семантический поиск (RAG)" с индикатором статуса
- Кнопка скачивания `nomic-embed-text` прямо из Settings
- Показывается статус: установлена / не установлена
- Прогресс скачивания с состоянием кнопки

### 🔵 V3 (ВИЗУАЛЬНЫЙ): 404 endpoint-ошибка показывалась как "Дела не найдены"
**Файл:** `src/pages/ClientDetails/components/ClientInfoPanel/KadArbitrDetails.tsx`

- Разделены случаи: техническая ошибка запроса vs реально пустой результат
- Правильные иконки и тексты для каждого типа ошибки
- Кнопка "Загрузить вручную" стала primary (основной workaround для Cloudflare)
- Отдельная обработка: Cloudflare 403, таймаут, endpoint-ошибка, пустой результат

---

## Что НЕ входит в этот патч (требует отдельной работы)

- **БАГ 4**: vec_embeddings захардкожен на FLOAT[768] — требует миграции БД
- **БАГ 6 (Cloudflare)**: reqwest не может обойти JS-Challenge — нужен WebView или прокси-API
- **V4**: color-mix() fallback для Windows 10 WebView2
- **V5**: единый источник selectedModel для всех агентов

## Как применить

1. Скопировать файлы из архива поверх соответствующих файлов проекта
2. Пересобрать Rust-бэкенд: `cargo build --release` (в `src-tauri/`)
3. Пересобрать фронтенд: `pnpm build`
4. В агентах (DocumentClassifierAgent и др.) заменить `AI_MODELS.widget` → `getActiveChatModel()`
