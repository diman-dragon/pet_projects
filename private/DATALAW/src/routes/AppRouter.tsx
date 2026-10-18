import React, { Suspense, useEffect, useState } from 'react';
import { Routes, Route, useLocation, Navigate } from 'react-router-dom';
import { FeatureGuard } from '../components/common/FeatureGuard';
import { PermissionGuard } from '../components/common/PermissionGuard';
import { PageLoader } from '../components/ui';
import { pathToStatus } from '../navigation/routes';
import type { NavigationID } from '../pages/clients/types';
import { ConfigService } from '../services/config/ConfigService';
import type { UserProfile } from '../types/entities';
import { PageWrapper } from './PageWrapper';

const Dashboard = React.lazy(() => import('../pages/dashboard/Dashboard'));
const ClientsPage = React.lazy(() => import('../pages/clients/ClientsPage'));
const UsersPage = React.lazy(() => import('../pages/users/UsersPage'));
const AuditPage = React.lazy(() => import('../pages/audit/Audit'));
const KnowledgeHub = React.lazy(() => import('../pages/knowledge/KnowledgeHub'));
const KnowledgeGraphPage = React.lazy(() => import('../pages/knowledgeGraph/KnowledgeGraphPage'));
const LailaPage = React.lazy(() => import('../pages/laila/LailaPage'));
const SystematizerPage = React.lazy(() => import('../pages/systematizer/SystematizerPage'));
const SettingsPage = React.lazy(() => import('../pages/settings/SettingsPage'));
const TrashPage = React.lazy(() => import('../pages/trash/TrashPage'));

interface AppRouterProps {
  status: string;
  currentUser: UserProfile | null;
  activeClientId: NavigationID;
  activeCaseId: NavigationID;
  activeLineId: NavigationID;
  activeAppealId: NavigationID;
  setStatus: (status: string) => void;
  setActiveClientId: (id: NavigationID) => void;
  setActiveCaseId: (id: NavigationID) => void;
  setActiveLineId: (id: NavigationID) => void;
  setActiveAppealId: (id: NavigationID) => void;
}

export function AppRouter({
  status,
  currentUser,
  activeClientId,
  activeCaseId,
  activeLineId,
  activeAppealId,
  setStatus,
  setActiveClientId,
  setActiveCaseId,
  setActiveLineId,
  setActiveAppealId,
}: AppRouterProps) {
  const [workDir, setWorkDir] = useState<string | null>(null);
  const location = useLocation();

  useEffect(() => {
    ConfigService.load().then((path) => setWorkDir(path ?? ''));
  }, []);

  useEffect(() => {
    const derivedStatus = pathToStatus(location.pathname);
    if (status !== derivedStatus) {
      setStatus(derivedStatus);
    }
  }, [location.pathname, setStatus, status]);

  if (status?.toLowerCase().trim() === 'loading' || status?.toLowerCase().trim() === 'checking') {
    return <PageLoader />;
  }

  return (
    <Suspense fallback={<PageLoader />}>
      <Routes>
        <Route
          path="/dashboard"
          element={
            <PageWrapper>
              <Dashboard />
            </PageWrapper>
          }
        />

        <Route
          path="/clients"
          element={
            <FeatureGuard feature="clients" fallback={<Navigate to="/dashboard" replace />}>
              <PermissionGuard
                user={currentUser}
                permission="clients:view"
                fallback={<Navigate to="/dashboard" replace />}
              >
                <PageWrapper>
                  <ClientsPage
                    activeClientId={activeClientId}
                    activeCaseId={activeCaseId}
                    activeLineId={activeLineId}
                    activeAppealId={activeAppealId}
                    setActiveClientId={setActiveClientId}
                    setActiveCaseId={setActiveCaseId}
                    setActiveLineId={setActiveLineId}
                    setActiveAppealId={setActiveAppealId}
                  />
                </PageWrapper>
              </PermissionGuard>
            </FeatureGuard>
          }
        />

        <Route
          path="/users"
          element={
            <FeatureGuard feature="users" fallback={<Navigate to="/dashboard" replace />}>
              <PermissionGuard
                user={currentUser}
                permission="users:view"
                fallback={<Navigate to="/dashboard" replace />}
              >
                <PageWrapper>
                  <UsersPage currentUser={currentUser} />
                </PageWrapper>
              </PermissionGuard>
            </FeatureGuard>
          }
        />

        <Route
          path="/audit"
          element={
            <FeatureGuard feature="audit" fallback={<Navigate to="/dashboard" replace />}>
              <PermissionGuard
                user={currentUser}
                permission="audit:view"
                fallback={<Navigate to="/dashboard" replace />}
              >
                {workDir === null ? (
                  <PageLoader />
                ) : (
                  <PageWrapper>
                    <AuditPage rootPath={workDir} />
                  </PageWrapper>
                )}
              </PermissionGuard>
            </FeatureGuard>
          }
        />

        <Route
          path="/knowledge"
          element={
            <FeatureGuard feature="knowledge" fallback={<Navigate to="/dashboard" replace />}>
              <PermissionGuard
                user={currentUser}
                permission="knowledge:view"
                fallback={<Navigate to="/dashboard" replace />}
              >
                <PageWrapper>
                  <KnowledgeHub />
                </PageWrapper>
              </PermissionGuard>
            </FeatureGuard>
          }
        />

        <Route
          path="/knowledge-graph"
          element={
            <FeatureGuard feature="knowledge" fallback={<Navigate to="/dashboard" replace />}>
              <PermissionGuard
                user={currentUser}
                permission="knowledge:view"
                fallback={<Navigate to="/dashboard" replace />}
              >
                <PageWrapper>
                  <KnowledgeGraphPage />
                </PageWrapper>
              </PermissionGuard>
            </FeatureGuard>
          }
        />

        <Route
          path="/laila"
          element={
            <FeatureGuard feature="laila" fallback={<Navigate to="/dashboard" replace />}>
              <PermissionGuard
                user={currentUser}
                permission="ai:use"
                fallback={<Navigate to="/dashboard" replace />}
              >
                <PageWrapper>
                  <LailaPage />
                </PageWrapper>
              </PermissionGuard>
            </FeatureGuard>
          }
        />

        <Route
          path="/systematizer"
          element={
            <FeatureGuard feature="laila" fallback={<Navigate to="/dashboard" replace />}>
              <PermissionGuard
                user={currentUser}
                permission="ai:use"
                fallback={<Navigate to="/dashboard" replace />}
              >
                <PageWrapper>
                  <SystematizerPage />
                </PageWrapper>
              </PermissionGuard>
            </FeatureGuard>
          }
        />

        <Route
          path="/settings"
          element={
            <FeatureGuard feature="settings" fallback={<Navigate to="/dashboard" replace />}>
              <PermissionGuard
                user={currentUser}
                permission="settings:view"
                fallback={<Navigate to="/dashboard" replace />}
              >
                <PageWrapper>
                  <SettingsPage />
                </PageWrapper>
              </PermissionGuard>
            </FeatureGuard>
          }
        />

        <Route
          path="/trash"
          element={
            <PermissionGuard
              user={currentUser}
              permission="clients:view"
              fallback={<Navigate to="/dashboard" replace />}
            >
              <PageWrapper>
                <TrashPage currentUser={currentUser} />
              </PageWrapper>
            </PermissionGuard>
          }
        />

        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Suspense>
  );
}
