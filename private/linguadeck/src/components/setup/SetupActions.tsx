import { Button } from "../../ui";

interface SetupActionsProps {
  busy: boolean;
  canInitialize: boolean;
  onInitialize: () => void;
}

export function SetupActions({ busy, canInitialize, onInitialize }: SetupActionsProps) {
  return (
    <Button
      variant="primary"
      size="lg"
      className="mt-6 w-full"
      disabled={!canInitialize || busy}
      isLoading={busy}
      onClick={onInitialize}
    >
      Запустить Linguadeck
    </Button>
  );
}
