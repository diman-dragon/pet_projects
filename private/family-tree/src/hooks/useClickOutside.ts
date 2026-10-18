import { useEffect } from "react";
import type { RefObject } from "react";

export function useClickOutside(
  elementRef: RefObject<HTMLElement | null>,
  onOutsideClick: () => void,
) {
  useEffect(() => {
    function handlePointerDown(event: PointerEvent) {
      const target = event.target;
      if (!(target instanceof Node) || elementRef.current?.contains(target)) {
        return;
      }
      onOutsideClick();
    }

    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [elementRef, onOutsideClick]);
}
