const APP_NAVIGATE_EVENT = 'Dorlane:navigate';

export interface AppNavigationRequest {
  route: string;
  params?: Record<string, unknown>;
}

export function requestAppNavigation(route: string, params?: Record<string, unknown>) {
  window.dispatchEvent(
    new CustomEvent<AppNavigationRequest>(APP_NAVIGATE_EVENT, {
      detail: { route, params },
    }),
  );
}

export function subscribeAppNavigation(
  listener: (request: AppNavigationRequest) => void,
): () => void {
  const handler = (event: Event) => {
    const custom = event as CustomEvent<AppNavigationRequest>;
    listener(custom.detail);
  };

  window.addEventListener(APP_NAVIGATE_EVENT, handler);
  return () => window.removeEventListener(APP_NAVIGATE_EVENT, handler);
}
