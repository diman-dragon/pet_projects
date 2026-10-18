import { Plus } from "lucide-react";
import { Button } from "../../ui/Button";
import { SearchField } from "../../ui/SearchField";

interface DecksToolbarProps {
  search: string;
  decksCount: number;
  onChangeSearch: (value: string) => void;
  onCreateDeck: () => void;
}

export function DecksToolbar({
  search,
  decksCount,
  onChangeSearch,
  onCreateDeck,
}: DecksToolbarProps) {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <SearchField
        value={search}
        onChange={onChangeSearch}
        placeholder="Поиск по названию колоды..."
        className="flex-1 min-w-[200px]"
      />
      <div className="h-6 w-px bg-gray-200" />
      <div className="text-sm text-gray-500">
        {decksCount} {decksCount === 1 ? "колода" : "колод"}
      </div>
      <div className="h-6 w-px bg-gray-200" />
      <Button onClick={onCreateDeck} variant="primary">
        <Plus size={16} />
        Создать колоду
      </Button>
    </div>
  );
}
