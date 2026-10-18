import type { ReactNode } from 'react';

interface PageWrapperProps {
  children: ReactNode;
}

export function PageWrapper({ children }: PageWrapperProps) {
  return <div className="h-full w-full overflow-y-auto scrollbar-hide">{children}</div>;
}
