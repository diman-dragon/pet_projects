// src/App.tsx
//
// P1-1 FIX: Добавлен MemoryRouter.
// MemoryRouter — правильный выбор для Tauri-десктопа:
//   - нет адресной строки браузера
//   - нет сервера, нет hash-роутинга
//   - история маршрутов живёт в памяти процесса
//   - BrowserRouter и HashRouter здесь неуместны

import { MemoryRouter } from 'react-router-dom';
import AppProviders from './core/AppProviders';
import AppGuards from './core/AppGuards';
import AppContent from './core/AppContent';
import { useAppState } from './core/hooks/useAppState';

function AppInner() {
  const appState = useAppState();
  return (
    <AppGuards appState={appState}>
      <AppContent appState={appState} />
    </AppGuards>
  );
}

export default function App() {
  return (
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <AppProviders>
        <AppInner />
      </AppProviders>
    </MemoryRouter>
  );
}
