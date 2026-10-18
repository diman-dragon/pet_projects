import { GrammarUnitDetail } from "./grammar/GrammarUnitDetail";
import { useGrammarPage } from "./hooks/useGrammarPage";
import { GrammarHeader } from "./grammar/GrammarHeader";
import { GrammarDashboard } from "./grammar/GrammarDashboard";
import { GrammarSidebar } from "./grammar/GrammarSidebar";

interface GrammarPageProps {
  focusUnitId?: number | null;
}

export function GrammarPage({ focusUnitId }: GrammarPageProps) {
  const {
    dashboard,
    units,
    selected,
    selectedId,
    setSelectedId,
    exercises,
    practice,
    answers,
    setAnswers,
    checked,
    setChecked,
    result,
    linkedCards,
    draft,
    setDraft,
    busy,
    importRef,
    createUnit,
    startPractice,
    finishPractice,
    exportSelected,
    importUnit,
    importDefaultModules,
  } = useGrammarPage(focusUnitId);

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 overflow-hidden">
      <GrammarHeader
        importRef={importRef}
        onImport={(file) => importUnit(file).catch(() => undefined)}
        onExport={exportSelected}
        onImportDefault={importDefaultModules}
        exportDisabled={!selected}
      />

      <GrammarDashboard dashboard={dashboard} />

      <div className="grid min-h-0 flex-1 grid-cols-[320px_1fr] gap-4">
        <GrammarSidebar
          units={units}
          selectedId={selectedId}
          onSelect={setSelectedId}
          draft={draft}
          onDraftChange={setDraft}
          onAdd={createUnit}
          busy={busy}
        />

        <section className="custom-scrollbar min-h-0 overflow-auto rounded-lg bg-white p-5 shadow-sm ring-1 ring-slate-200">
          {selected ? (
            <GrammarUnitDetail
              selected={selected}
              practice={practice}
              exercises={exercises}
              answers={answers}
              checked={checked}
              result={result}
              linkedCards={linkedCards}
              onStartPractice={startPractice}
              onFinishPractice={finishPractice}
              onAnswer={(id, value) => setAnswers((cur) => ({ ...cur, [id]: value }))}
              onCheck={(id) => setChecked((cur) => ({ ...cur, [id]: true }))}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-slate-500">
              Создайте первую грамматическую тему.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
