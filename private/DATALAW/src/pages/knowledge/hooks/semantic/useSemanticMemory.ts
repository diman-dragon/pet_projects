import { useCallback, useRef, useState } from 'react';

export type MemoryMessage = { role: 'user' | 'assistant'; content: string };

export function useSemanticMemory() {
  const [memoryMessages, setMemoryMessages] = useState<MemoryMessage[]>([]);
  const memoryRef = useRef<MemoryMessage[]>([]);

  const addToMemory = useCallback((role: 'user' | 'assistant', content: string) => {
    setMemoryMessages((prev) => {
      const next = [...prev.slice(-49), { role, content }];
      memoryRef.current = next;
      return next;
    });
  }, []);

  const clearMemory = useCallback(() => {
    memoryRef.current = [];
    setMemoryMessages([]);
  }, []);

  return {
    memoryMessages,
    memoryRef,
    addToMemory,
    clearMemory,
  };
}
