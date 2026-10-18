// src/core/AppProviders.tsx
import { ReactNode } from 'react';
import { AppLanguageProvider } from '../i18n/AppLanguageProvider';

export default function AppProviders({ children }: { children: ReactNode }) {
  return <AppLanguageProvider>{children}</AppLanguageProvider>;
}
