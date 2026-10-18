# Dorlane — Исправления семантического поиска

## Изменённые файлы

```
src/services/ai/embeddingModels.ts
src/services/ai/agents/KnowledgeSearchAgent.ts
src/services/rag/EmbeddingService.ts
src/services/rag/VectorSearchService.ts
src/services/rag/SearchPipeline.ts
src/services/rag/embedding/local.ts
src/services/rag/search/cosine.ts
src/services/rag/search/hybrid.ts
src/services/rag/indexer/chunkWriter.ts
src/services/rag/indexer/forceReembedAll.ts
```

---

## БАГ 1 — Дефолтная модель изменена на bge-m3:latest

**Файл:** `src/services/ai/embeddingModels.ts`

- `DEFAULT_EMBEDDING_MODEL` = `'bge-m3:latest'` (было `nomic-embed-text:latest`)
- BGE-M3 многоязычная (русский + английский), 1024 измерений
- Добавлен `promptStyle: 'bge'` — BGE-M3 не нужны task prefix'ы

**ВАЖНО:** После обновления нужно переиндексировать базу знаний
(кнопка «Переиндексировать» в настройках), т.к. старые эмбеддинги
сделаны другой моделью и несовместимы.

---

## БАГ 2 — EmbeddingService: убрана зависимость от Tauri IPC

**Файл:** `src/services/rag/EmbeddingService.ts`

Раньше `normalizeVector()` и `cosineSimilarity()` вызывали
`invokeRagCommand()` → Tauri IPC. Если Rust-плагин не ответил —
всё молча падало, векторы не создавались.

**Исправление:** обе функции переписаны на чистый JS. Нет зависимости от Tauri.

---

## БАГ 3 — embedding/local.ts: добавлен BGE prompt style

**Файл:** `src/services/rag/embedding/local.ts`

- Добавлена ветка `case 'bge'` — без prefix (BGE-M3 не нужны)
- Hash fallback теперь использует правильную размерность для каждой модели
  (BGE-M3 = 1024 dim, nomic = 768 dim)

---

## БАГ 4 — searchJsCosine: убрана зависимость от Tauri IPC

**Файл:** `src/services/rag/search/cosine.ts`

`scoreVectorBatch()` вызывал `invokeRagCommand('rag_score_vectors')` — Tauri IPC.
При любой ошибке catch возвращал `[]`, пользователь видел пустой результат.

**Исправление:** заменён на `dotProduct()` в чистом JS.
Векторы нормализованы при сохранении, поэтому dot product = cosine similarity.

---

## БАГ 5 — DISABLE_FULLTEXT_FALLBACK = true

**Файлы:** `VectorSearchService.ts`, `SearchPipeline.ts`

Флаг отключал FTS. Если векторный поиск падал → 0 результатов.

**Исправление:** флаг удалён. FTS всегда работает как резерв.
В `SearchPipeline.ts` шаг 3 (FTS) выполняется всегда независимо от результатов вектора.

---

## БАГ 6 — hybrid.ts: Tauri fallback

**Файл:** `src/services/rag/search/hybrid.ts`

`mergeHybridResults()` тоже вызывал Tauri IPC. При ошибке мёрж не происходил.

**Исправление:** добавлен `mergeHybridJS()` — чистый JS fallback.
Если Tauri IPC бросает ошибку — автоматически используется JS версия.

---

## БАГ 7 — getCompatibleEmbeddingModels: возвращал [] при ошибке

**Файл:** `VectorSearchService.ts`

Теперь при base-match возвращает все найденные варианты модели из БД.
Логирует детально что нашлось и почему.

---

## КАК ПРОВЕРИТЬ ЧТО ВСЁ РАБОТАЕТ

1. Запустите `ollama pull bge-m3:latest`
2. Откройте приложение → База знаний → Настройки → Переиндексировать
3. Подождите индексации
4. Выполните поисковый запрос (минимум 2 слова)
5. В DevTools Console должно быть:
   ```
   [VectorSearchService] embeddingMode: real
   [Search] Vector results: N
   [Search] FTS results: N
   [Pipeline] Merged: N
   ```

### Признаки правильной работы
- `embeddingMode: "real"` (не "hash")  
- `Vector results` > 0
- `compatibleModels: ["bge-m3:latest"]`

### Признаки проблемы
- `embeddingMode: "hash"` → Ollama недоступна или модель не скачана
- `No compatible embeddings` → нужна переиндексация
