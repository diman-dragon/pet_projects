// src/components/common/FeatureGuard.tsx
//
// Обёртка для условного рендера по наличию фичи в лицензии.
// Пример:
//   <FeatureGuard feature="rag" fallback={<UpgradeBanner />}>
//     <KnowledgeSearch />
//   </FeatureGuard>

import type { ReactNode } from 'react';
import type { FeatureName } from '../../services/license/config';
import { useLicenseStore } from '../../store/useLicenseStore';

interface FeatureGuardProps {
  feature: FeatureName;
  /** Что показать, если фича недоступна. По умолчанию — ничего. */
  fallback?: ReactNode;
  children: ReactNode;
}

export function FeatureGuard({ feature, fallback = null, children }: FeatureGuardProps) {
  const hasFeature = useLicenseStore((state) => state.hasFeature(feature));
  return hasFeature ? <>{children}</> : <>{fallback}</>;
}
