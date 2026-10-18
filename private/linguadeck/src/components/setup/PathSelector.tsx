import { Button } from "../../ui";

interface PathSelectorProps {
  selectedPath: string | null;
  onSelect: () => void;
}

export function PathSelector({ selectedPath, onSelect }: PathSelectorProps) {
  return (
    <div className="mt-8 rounded-[24px] border border-dashed border-slate-300 bg-slate-50/80 p-6 text-center">
      <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-slate-400 mb-4">
        Library Path
      </div>
      <div className="min-h-14 rounded-2xl bg-white px-4 py-4 text-sm font-medium text-slate-700 shadow-inner border border-slate-100">
        {selectedPath ?? "Папка еще не выбрана"}
      </div>
      <Button 
        variant="secondary" 
        className="mt-4 w-full" 
        onClick={onSelect}
      >
        Выбрать папку
      </Button>
    </div>
  );
}
