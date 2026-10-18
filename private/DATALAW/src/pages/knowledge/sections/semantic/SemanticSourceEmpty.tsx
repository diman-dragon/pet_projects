import { ChevronDown } from 'lucide-react';

export function SemanticSourceEmpty({ text }: { text: string }) {
  return (
    <div className="flex h-full flex-col items-center justify-center space-y-3 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-black/[0.03]">
        <ChevronDown className="h-5 w-5 text-black/30" />
      </div>
      <p className="max-w-[220px] text-xs leading-relaxed text-black/40">{text}</p>
    </div>
  );
}
