// hooks/useAutoScroll.ts

import { RefObject, useEffect } from 'react';

/**
 * Автоматически прокручивает `ref` вниз при изменении `deps`.
 * @param ref    - ref на элемент-якорь (bottomRef)
 * @param deps   - зависимости (обычно [messages, isThinking])
 * @param instant - использовать мгновенный скролл при монтировании
 */
export function useAutoScroll(ref: RefObject<HTMLElement>, deps: unknown[], instant = false) {
  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: instant ? 'instant' : 'smooth' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
