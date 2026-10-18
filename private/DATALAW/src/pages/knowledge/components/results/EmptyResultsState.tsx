import { SearchX, Database } from 'lucide-react';

interface EmptyResultsStateProps {
  hasIndex: boolean;
}

export function EmptyResultsState({ hasIndex }: EmptyResultsStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-10 text-center px-4">
      <div className="mb-3 flex size-10 items-center justify-center rounded-2xl bg-black/[0.04]">
        {hasIndex
          ? <SearchX size={18} className="text-black/25" />
          : <Database size={18} className="text-black/25" />
        }
      </div>
      <p className="text-[length:var(--text-xs)] font-semibold text-black/40">
        {hasIndex ? 'Ничего не найдено' : 'База знаний пуста'}
      </p>
      <p className="mt-1.5 text-[10px] text-black/25 leading-relaxed max-w-[200px]">
        {hasIndex
          ? 'Попробуйте другие слова или уточните запрос'
          : 'Перейдите в «Библиотека» и проиндексируйте документы'
        }
      </p>
    </div>
  );
}
