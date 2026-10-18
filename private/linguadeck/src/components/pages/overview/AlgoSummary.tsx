import { Card } from "../../../ui";

interface AlgoSummaryProps {
  algoName: string;
  steps: string;
  limit: number;
}

export function AlgoSummary({ algoName, steps, limit }: AlgoSummaryProps) {
  return (
    <Card className="p-4 flex justify-between items-center">
      <div>
        <div className="text-xs text-slate-500 uppercase font-bold tracking-wider">FSRS</div>
        <div className="font-bold text-slate-900 mt-1">Алгоритм: {algoName}</div>
      </div>
      <div className="text-right">
        <div className="text-sm text-slate-600">Шаги: {steps}</div>
        <div className="text-sm text-slate-600">Лимит сессии: {limit} карт</div>
      </div>
    </Card>
  );
}
