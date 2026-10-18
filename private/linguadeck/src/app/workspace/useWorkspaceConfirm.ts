import { useState } from "react";

export type WorkspaceConfirmState =
  | { kind: "card"; id: number }
  | { kind: "cards" }
  | { kind: "deck"; id: number }
  | null;

export function useWorkspaceConfirm() {
  const [confirmState, setConfirmState] = useState<WorkspaceConfirmState>(null);

  function closeConfirm() {
    setConfirmState(null);
  }

  return {
    confirmState,
    setConfirmState,
    closeConfirm
  };
}
