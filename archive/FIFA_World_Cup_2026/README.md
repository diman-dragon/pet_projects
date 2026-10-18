# ⚽ FIFA World Cup 2026 Dashboard & Stats Engine

[Русский](#русский) | [English](#english)

---

<a name="русский"></a>
## 🇷🇺 Описание проекта

Этот проект представляет собой интеллектуальную систему сбора, нормализации и визуализации данных о Чемпионате Мира по футболу 2026 года. Система спроектирована как полностью автономный инструмент, превращающий разрозненные данные ESPN API в структурированную, аналитическую базу знаний для болельщиков и аналитиков.

### 🛠 Технический стек
*   **Backend:** Python 3.x (автоматизация сбора данных, обработка JSON, нормализация статистики).
*   **Data Source:** ESPN Soccer API (футбольная аналитика в реальном времени).
*   **Frontend:** Vanilla JS, CSS (быстрый, легковесный интерфейс без лишних зависимостей).
*   **Deployment:** Локальный Python HTTP-сервер для мгновенного доступа к статистике.

### 🏛 Архитектура и поток данных
Система работает по конвейерному принципу:
```mermaid
graph TD
    A[ESPN API] --> B(loader.py - Расписание);
    A --> C(loader_stats.py - Статистика матчей);
    A --> D(ingest_data.py - Pro-данные);
    B --> E[js/data/*.js (JSONP)];
    C --> E;
    D --> E;
    E --> F[Web Frontend (Vanilla JS)];
```

### ⚙️ Стратегия нормализации данных
Ключевая особенность проекта — **унификация имен**. ESPN API отдает названия команд на английском, в то время как интерфейс требует отображения на русском.
*   Используется централизованный словарь `TEAM_NAME_MAP`.
*   Все скрипты сборки (`loader.py`, `loader_stats.py`, `ingest_data.py`) проходят через функцию `normalize_team()`, что гарантирует полное соответствие данных в разных JS-файлах.

### 📂 Файловый манифест
*   **Скрипты (Backend):**
    *   `loader.py`: Сбор расписания и базовой информации о матчах за 4 дня.
    *   `loader_stats.py`: Агрегация детализированной статистики (владение, карточки, фолы).
    *   `ingest_data.py`: Сбор про-данных (boxscores).
*   **Данные (Frontend):**
    *   `js/data/tournament_data.js`: Основной файл расписания.
    *   `js/data/full_match_stats.js`: Агрегированная статистика.
    *   `js/data/final/tournament_data_pro.js`: Глубокие данные для аналитики.

### 🚀 Инструкция по развертыванию
1. **Предварительные требования:** Установленный Python 3.
2. **Запуск:** Запустите `Старт.bat` из корня проекта.
3. **Управление:** В консольном меню выберите:
    *   `1`: Обновление всех данных (расписание + статистика).
    *   `2`: Запуск сервера.
    *   `3`: Полный цикл обновления и запуска.
4. **Доступ:** Перейдите по адресу `http://localhost:8000`.

---

<a name="english"></a>
## 🇬🇧 Project Description

This project is an intelligent data collection, normalization, and visualization engine for the 2026 FIFA World Cup. It is designed as a fully autonomous tool that transforms raw ESPN API data into a structured, analytical knowledge base for fans and analysts.

### 🛠 Tech Stack
*   **Backend:** Python 3.x (data automation, JSON processing, statistical normalization).
*   **Data Source:** ESPN Soccer API (real-time football analytics).
*   **Frontend:** Vanilla JS, CSS (fast, lightweight, dependency-free interface).
*   **Deployment:** Local Python HTTP server for instant access.

### 🏛 Architecture & Data Flow
The system operates as a data pipeline:
```mermaid
graph TD
    A[ESPN API] --> B(loader.py - Schedule);
    A --> C(loader_stats.py - Match Stats);
    A --> D(ingest_data.py - Pro Data);
    B --> E[js/data/*.js (JSONP)];
    C --> E;
    D --> E;
    E --> F[Web Frontend (Vanilla JS)];
```

### ⚙️ Data Normalization Strategy
A key feature of this project is **name unification**. The ESPN API provides team names in English, while the interface requires Russian localization.
*   A centralized `TEAM_NAME_MAP` dictionary is utilized.
*   All collection scripts (`loader.py`, `loader_stats.py`, `ingest_data.py`) utilize the `normalize_team()` function, ensuring data consistency across disparate JS files.

### 📂 File Manifest
*   **Backend Scripts:**
    *   `loader.py`: Fetches schedule and basic match info for the last 4 days.
    *   `loader_stats.py`: Aggregates detailed statistics (possession, cards, fouls).
    *   `ingest_data.py`: Collects pro-level boxscore data.
*   **Frontend Data:**
    *   `js/data/tournament_data.js`: Core tournament schedule.
    *   `js/data/full_match_stats.js`: Aggregated match statistics.
    *   `js/data/final/tournament_data_pro.js`: Deep analytics data.

### 🚀 Deployment Instructions
1. **Prerequisites:** Python 3 installed.
2. **Execution:** Run `Старт.bat` from the project root.
3. **Control:** Choose from the console menu:
    *   `1`: Full data update (schedule + stats).
    *   `2`: Start local server.
    *   `3`: Full update and serve.
4. **Access:** Navigate to `http://localhost:8000`.
