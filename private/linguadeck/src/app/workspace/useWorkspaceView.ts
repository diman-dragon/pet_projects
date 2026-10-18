import { useState } from "react";
import type { ViewId } from "../../components/layout/SidebarNav";

export function useWorkspaceView() {
  const [view, setView] = useState<ViewId>("dashboard");

  return {
    view,
    setView
  };
}
