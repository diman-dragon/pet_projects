// src/pages/knowledge/hooks/useFavorites.ts

import { useState, useCallback } from 'react';

/**
 * Управляет избранными документами (Set<id>).
 * Возвращает: set, toggle, has-checker, размер.
 */
export function useFavorites() {
  const [favorites, setFavorites] = useState<Set<string>>(new Set());

  const toggleFavorite = useCallback((id: string) => {
    setFavorites((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }, []);

  const isFavorite = useCallback((id: string) => favorites.has(id), [favorites]);

  return { favorites, toggleFavorite, isFavorite };
}
