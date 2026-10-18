export const ROUTE_MAP = {
  dashboard: '/dashboard',
  clients: '/clients',
  users: '/users',
  staff: '/users',
  audit: '/audit',
  knowledge: '/knowledge',
  knowledge_graph: '/knowledge-graph',
  laila: '/laila',
  systematizer: '/systematizer',
  settings: '/settings',
  trash: '/trash',
  'ai-control': '/settings',
} as const;

export type AppRouteStatus = keyof typeof ROUTE_MAP;

export function normalizeRouteStatus(value: string | null | undefined): AppRouteStatus {
  const key = value?.toLowerCase().trim() as AppRouteStatus | undefined;
  return key && key in ROUTE_MAP ? key : 'dashboard';
}

export function statusToPath(value: string | null | undefined): string {
  return ROUTE_MAP[normalizeRouteStatus(value)];
}

export function pathToStatus(pathname: string | null | undefined): AppRouteStatus {
  const normalized = pathname?.toLowerCase().trim() || '';
  const match = Object.entries(ROUTE_MAP).find(([, path]) => path === normalized);
  return (match?.[0] as AppRouteStatus | undefined) ?? 'dashboard';
}
