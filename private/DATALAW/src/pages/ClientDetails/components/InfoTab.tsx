// src/pages/ClientDetails/components/InfoTab.tsx
import { Client, Case } from '../../../types/entities';
import { CasesStats } from './CasesStats';
import { ClientInfoPanel } from './ClientInfoPanel';

interface Props {
  client: Client;
  cases: Case[];
  clientId: number;
  isSummaryGenerating?: boolean;
  onGenerateSummary?: () => Promise<void>;
  isRiskScanning?: boolean;
  criticalRisksCount?: number;
  onScanRisks?: () => Promise<void>;
  groupedRisks?: {
    critical: Array<{ description: string; suggestedAction?: string }>;
    warnings: Array<{ description: string; suggestedAction?: string }>;
    ok: string[];
  };
}

export function InfoTab({
  client,
  cases,
  clientId,
  isSummaryGenerating,
  onGenerateSummary,
  isRiskScanning,
  criticalRisksCount,
  onScanRisks,
  groupedRisks,
}: Props) {
  return (
    <div className="flex flex-col h-full animate-in fade-in slide-in-from-bottom-2 duration-700 overflow-y-auto custom-scrollbar">
      <div className="px-10 pt-10 pb-4">
        <CasesStats cases={cases} />
      </div>

      <div className="px-10 py-6 max-w-[1600px]">
        <ClientInfoPanel
          client={client}
          clientId={clientId}
          isSummaryGenerating={isSummaryGenerating}
          onGenerateSummary={onGenerateSummary}
          isRiskScanning={isRiskScanning}
          criticalRisksCount={criticalRisksCount}
          onScanRisks={onScanRisks}
          groupedRisks={groupedRisks}
        />
      </div>
    </div>
  );
}
