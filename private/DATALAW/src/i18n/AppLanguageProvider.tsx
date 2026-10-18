import { createContext, useCallback, useContext, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import type { TFunction } from 'i18next';
import { AppLanguageService } from '../services/config/AppLanguageService';
import type { AppLanguage } from './config';
import i18n, { normalizeAppLanguage } from './i18n';
import type { TranslationKey } from './translations';

interface AppLanguageContextValue {
  language: AppLanguage;
  setLanguage: (language: AppLanguage) => Promise<void>;
  resetLanguageToSystem: () => Promise<void>;
  t: (key: TranslationKey, params?: Record<string, string | number>) => string;
}

const AppLanguageContext = createContext<AppLanguageContextValue | null>(null);

export function AppLanguageProvider({ children }: { children: React.ReactNode }) {
  const { t: i18nT } = useTranslation('common');
  const tFn = i18nT as TFunction<'common', undefined>;
  const language = normalizeAppLanguage(i18n.language);
  const syncedRef = useRef(false);

  useEffect(() => {
    if (syncedRef.current) return;
    syncedRef.current = true;
    void AppLanguageService.get().then((resolved) => {
      const next = normalizeAppLanguage(resolved);
      if (normalizeAppLanguage(i18n.language) !== next) {
        void i18n.changeLanguage(next);
      }
    });
  }, []);

  const setLanguage = useCallback(async (nextLanguage: AppLanguage) => {
    await i18n.changeLanguage(nextLanguage);
    await AppLanguageService.set(nextLanguage);
  }, []);

  const resetLanguageToSystem = useCallback(async () => {
    const detected = AppLanguageService.getSystemLanguage();
    await i18n.changeLanguage(detected);
    await AppLanguageService.resetToSystem();
  }, []);

  const t = useCallback(
    (key: TranslationKey, params?: Record<string, string | number>) =>
      (tFn as (k: string, opts?: Record<string, string | number>) => string)(key, params),
    [tFn],
  );

  const value = useMemo<AppLanguageContextValue>(
    () => ({
      language,
      setLanguage,
      resetLanguageToSystem,
      t,
    }),
    [language, setLanguage, resetLanguageToSystem, t],
  );

  return <AppLanguageContext.Provider value={value}>{children}</AppLanguageContext.Provider>;
}

export function useI18n() {
  const context = useContext(AppLanguageContext);
  if (!context) {
    throw new Error('useI18n must be used within AppLanguageProvider');
  }
  return context;
}
