// src/main.tsx

import React from 'react';
import ReactDOM from 'react-dom/client';
import './i18n/i18n';
import App from './App';
import './index.css';
import { patchConsole } from './services/logger/Logger';
import { ErrorReporter } from './services/errors/ErrorReporter';

// Инициализируем перехват логов и глобальные обработчики ошибок
patchConsole();
ErrorReporter.installGlobalHandlers();

/**
 * Точка входа в UI-слой проекта Dorlane.
 * Используем StrictMode для обеспечения стабильности и выявления
 * устаревших методов жизненного цикла компонентов.
 */
ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
