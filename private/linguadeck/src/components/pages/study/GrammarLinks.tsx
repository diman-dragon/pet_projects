interface GrammarLinksProps {
  links: Array<{ id: number; title: string }>;
  onOpen: (id: number) => void;
}

export function GrammarLinks({ links, onOpen }: GrammarLinksProps) {
  if (links.length === 0) return null;

  return (
    <div className="mx-5 mb-2 flex flex-wrap items-center justify-center gap-2">
      {links.map((unit) => (
        <button
          key={unit.id}
          onClick={() => onOpen(unit.id)}
          className="rounded-full bg-blue-50 px-3 py-1 text-xs font-bold text-blue-700 ring-1 ring-blue-100 hover:bg-blue-100"
          type="button"
        >
          {unit.title}
        </button>
      ))}
    </div>
  );
}
