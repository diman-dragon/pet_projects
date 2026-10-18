// src/pages/LineDetails/components/InfoTab.tsx
import { Line, Appeal } from '../../../types/entities';
import { AppealsStats } from './AppealsStats';
import { LineInfoPanel } from './LineInfoPanel';

interface Props {
  line: Line;
  appeals: Appeal[];
}

export function InfoTab({ line, appeals }: Props) {
  return (
    <div className="flex flex-col h-full animate-in fade-in slide-in-from-bottom-2 duration-500 overflow-y-auto scrollbar-thin">
      {/* Статистика: теперь она ТОЛЬКО здесь, как ты и просил */}
      <div className="px-8 pt-6 pb-2">
        <AppealsStats appeals={appeals} />
      </div>

      {/* Основная панель с данными линии */}
      <div className="max-w-5xl">
        <LineInfoPanel line={line} />
      </div>
    </div>
  );
}
