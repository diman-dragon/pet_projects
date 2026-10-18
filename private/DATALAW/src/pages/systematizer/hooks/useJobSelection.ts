import { useEffect, useMemo, useState } from 'react';

export function useJobSelection(ids: string[]) {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  // Cleanup removed IDs (use JSON.stringify to avoid infinite loop when array reference changes)
  useEffect(() => {
    setSelectedIds((prev) => prev.filter((id) => ids.includes(id)));
  }, [JSON.stringify(ids)]);

  const toggle = (id: string) =>
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]));

  const reset = () => setSelectedIds([]);

  const allSelected = useMemo(
    () => ids.length > 0 && ids.every((id) => selectedIds.includes(id)),
    [ids, selectedIds],
  );

  const toggleAll = () => setSelectedIds(allSelected ? [] : ids);

  return { selectedIds, isSelected: (id: string) => selectedIds.includes(id), toggle, toggleAll, allSelected, reset };
}
