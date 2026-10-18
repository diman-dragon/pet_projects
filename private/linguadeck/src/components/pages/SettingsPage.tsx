import { useState } from "react";
import type { AppLogEntry, AppSettings } from "../../types/domain";
import { AlgorithmSettings, LanguageSettings, LogsSection } from "../settings";

interface SettingsPageProps {
  settings: AppSettings;
  logs: AppLogEntry[];
  onChangeSettings: (patch: Partial<AppSettings>) => void;
  onClearLogs: () => void;
}

type SettingsTab = "language" | "algorithm" | "logs";

export function SettingsPage(props: SettingsPageProps) {
  const [activeTab] = useState<SettingsTab>("language");

  return (
    <section className="space-y-6">
      <h1 className="text-2xl font-bold text-slate-900">Настройки</h1>
      <p className="text-sm text-slate-500">Языки, алгоритм обучения и системные логи</p>

      <div className="flex justify-center">
        {/* Placeholder for SegmentButton removed or replaced */}
      </div>
      {activeTab === "language" ? (
        <LanguageSettings onChange={props.onChangeSettings} settings={props.settings} />
      ) : null}
      {activeTab === "algorithm" ? (
        <AlgorithmSettings onChange={props.onChangeSettings} settings={props.settings} />
      ) : null}
      {activeTab === "logs" ? <LogsSection logs={props.logs} onClearLogs={props.onClearLogs} /> : null}
    </section>
  );
}
