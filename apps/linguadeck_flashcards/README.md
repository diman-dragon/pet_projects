# LinguaDeck Flashcards 🎴✨

[![PWA Ready](https://img.shields.io/badge/PWA-Ready-success?style=flat-square&logo=pwa)](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)
[![SQLite WebAssembly](https://img.shields.io/badge/Database-SQLite%20WASM-blue?style=flat-square&logo=sqlite)](https://github.com/sql-js/sql.js)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

> A modern, ultra-fast, local-first web application for flashcard language learning powered by Spaced Repetition (SRS), SQLite in WebAssembly, and seamless Google Drive sync.

---

## 🌐 Language Navigation
- [English Documentation](#-english-version) *(Primary)*
- [Русская Документация](#-русская-версия)

---

<a name="-english-version"></a>
## 🇬🇧 English Version

### 🚀 Key Features

* **🧠 Spaced Repetition System (SRS):** Smart interval scheduling algorithm designed to optimize memory retention and reduce review time.
* **💾 Local-First & SQLite WASM:** Complete data ownership. Powered by `sql.js` (SQLite compiled to WebAssembly) running directly in your browser.
* **🔄 Google Drive Sync:** Effortlessly backup, restore, and synchronize your decks across device boundaries without relying on centralized third-party servers.
* **📦 Rich Media Support:** Attach audio pronunciations (`.mp3`) and visual cues (`.jpg`) to flashcards for multi-sensory retention.
* **📱 Progressive Web App (PWA):** Installable on Desktop, iOS, and Android. Full offline support via Service Workers.
* **⚡ Batch Add & CSV/JSON Import:** Quick workflows to import large vocabularies or create custom card bundles on the fly.
* **📊 Analytics & Feedback:** Interactive stats tracking your learning progress, mastery levels, and daily streak.

---

### 📂 Directory Structure

```text
linguadeck_flashcards/
├── css/                   # Modular design system (tokens, components, layouts)
├── js/                    # Core app mechanics
│   ├── app.js             # Entry point & bootstrap
│   ├── srs.js             # Spaced Repetition logic & algorithms
│   ├── db.js              # SQLite database adapter & query wrappers
│   ├── googleDriveSync.js # Google Drive cloud sync integration
│   ├── mediaStore.js      # Media blob caching & management
│   ├── study.js           # Interactive flashcard session controller
│   └── ...                # Modular UI dialogs & action handlers
├── media/                 # Default media resources (Capitals, English B1, etc.)
├── vendor/                # Standalone dependencies (sql-wasm.js, sql-wasm.wasm)
├── index.html             # Single-Page Application root
├── manifest.json          # PWA metadata configuration
└── service-worker.js      # Offline caching & service worker implementation
```

---

### 🛠️ Getting Started

Because LinguaDeck is **local-first** and runs directly in the browser, setting it up is simple.

#### 1. Quick Local Server
Clone the repository and serve it using any HTTP server:

```bash
git clone https://github.com/your-username/linguadeck_flashcards.git
cd linguadeck_flashcards

# Using Python 3
python -m http.server 8000

# OR using Node.js (npx)
npx http-server -p 8000
```
Open `http://localhost:8000` in your browser.

#### 2. Installing as PWA
1. Open the application in a modern browser (Chrome, Edge, Safari).
2. Click **Install App** or **Add to Home Screen** from your browser options.
3. Enjoy offline study sessions anytime!

---

### 📊 Tech Stack

* **Frontend:** Vanilla JS (ES6+), Modular CSS, HTML5
* **Database:** SQLite WebAssembly (`sql.js`)
* **Storage & Caching:** IndexedDB, Cache API, Service Workers
* **Integrations:** Google Drive REST API v3

---

<a name="-русская-версия"></a>
## 🇷🇺 Русская Версия

### 🚀 Основные Возможности

* **🧠 Интервальные повторения (SRS):** Умный алгоритм планирования повторений для максимального запоминания слов за минимальное время.
* **💾 Local-First & SQLite WASM:** Полный контроль над вашими данными. Использование `sql.js` (SQLite, скомпилированный в WebAssembly) прямо в браузере.
* **🔄 Синхронизация с Google Drive:** Простая архивация и синхронизация колод между устройствами без сторонних проприетарных серверов.
* **📦 Поддержка медиафайлов:** Прикрепляйте аудиопроизношение (`.mp3`) и иллюстрации (`.jpg`) к карточкам.
* **📱 Progressive Web App (PWA):** Возможность установки на ПК, iOS и Android. Полноценная работа в офлайн-режиме с помощью Service Worker.
* **⚡ Массовое добавление и импорт:** Быстрая загрузка словарей через встроенный пакетный ввод и импорт.
* **📊 Статистика и аналитика:** Отслеживание прогресса обучения, уровня усвоения материала и ежедневной активности.

---

### 🛠️ Быстрый Запуск

Так как приложение **LinguaDeck** работает прямо в браузере и не требует тяжелого бэкенда, запуск занимает секунды:

```bash
# Клонируйте репозиторий
git clone https://github.com/your-username/linguadeck_flashcards.git
cd linguadeck_flashcards

# Запустите локальный сервер (например, через Python)
python -m http.server 8000
```
Перейдите по адресу `http://localhost:8000` в вашем браузере.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.