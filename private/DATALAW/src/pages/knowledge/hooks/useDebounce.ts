// src/pages/knowledge/hooks/useDebounce.ts

import { useState, useEffect, useRef } from 'react';

/**
 * Возвращает дебаунсированное значение и флаг ожидания.
 * @param value - значение для дебаунса
 * @param delay - задержка в мс (по умолчанию 350)
 *
 * FIX: Убран безусловный setPending(true) из тела useEffect — он вызывал
 * бесконечный цикл setState → render → useEffect → setState.
 * Теперь pending выставляется только когда значение реально изменилось
 * (сравниваем через ref), а delay стабилизирован через ref чтобы изменение
 * числа-константы не перезапускало эффект.
 */
export function useDebounce<T>(value: T, delay = 350): [T, boolean] {
  const [debounced, setDebounced] = useState<T>(value);
  const [pending, setPending] = useState(false);
  const delayRef = useRef(delay);
  const prevValueRef = useRef<T>(value);

  // Обновляем ref задержки без перезапуска эффекта
  useEffect(() => {
    delayRef.current = delay;
  });

  useEffect(() => {
    // Если значение не изменилось — не начинаем ожидание
    if (Object.is(prevValueRef.current, value) && debounced === value) {
      return;
    }
    prevValueRef.current = value;
    setPending(true);

    const t = setTimeout(() => {
      setDebounced(value);
      setPending(false);
    }, delayRef.current);

    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return [debounced, pending];
}
