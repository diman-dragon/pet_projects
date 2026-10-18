// src/components/ui/feedback/StreamingCursor.tsx
// Canonical location. Imported by laila/chat and knowledge/shared.

/** Мигающий курсор, отображаемый во время стриминга ответа */
export function StreamingCursor() {
  return (
    <span className="inline-block w-[2px] h-[13px] bg-violet-400 ml-0.5 align-middle animate-pulse" />
  );
}
