import React from 'react';
import { SortAsc, SortDesc } from 'lucide-react';
import type { SortDir } from './utils';

interface SortButtonProps {
  active: boolean;
  dir: SortDir;
  onClick: () => void;
  children: string;
}

export const SortButton: React.FC<SortButtonProps> = ({ active, dir, onClick, children }) => (
  <button
    onClick={onClick}
    className={`h-7 flex items-center gap-1 px-2.5 rounded-md text-[length:var(--text-xs)] font-semibold border transition-all
            ${active ? 'border-black/20 bg-black text-white' : 'border-black/[0.08] text-black/40 hover:text-black/60 hover:bg-black/[0.03]'}`}
  >
    {children}
    {active && (dir === 'asc' ? <SortAsc size={10} /> : <SortDesc size={10} />)}
  </button>
);
