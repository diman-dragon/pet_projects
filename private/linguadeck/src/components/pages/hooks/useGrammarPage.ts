import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { englishGrammarModules } from "../../../data/englishGrammar";
import { GrammarService, type GrammarUnitInput } from "../../../services/db/GrammarService";
import type { Card, GrammarDashboard, GrammarExercise, GrammarExerciseType, GrammarSessionResult, GrammarUnit } from "../../../types/domain";
import { emptyUnit, demoExercises, normalize } from "../grammar/grammarHelpers";

export function useGrammarPage(focusUnitId?: number | null) {
  const [dashboard, setDashboard] = useState<GrammarDashboard | null>(null);
  const [units, setUnits] = useState<GrammarUnit[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [exercises, setExercises] = useState<GrammarExercise[]>([]);
  const [practice, setPractice] = useState<GrammarExercise[]>([]);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [checked, setChecked] = useState<Record<number, boolean>>({});
  const [result, setResult] = useState<GrammarSessionResult | null>(null);
  const [linkedCards, setLinkedCards] = useState<Card[]>([]);
  const [draft, setDraft] = useState<GrammarUnitInput>(emptyUnit);
  const [busy, setBusy] = useState(false);
  const importRef = useRef<HTMLInputElement>(null);

  const selected = useMemo(
    () => units.find((u) => u.id === selectedId) ?? units[0] ?? null,
    [units, selectedId]
  );

  const load = useCallback(async () => {
    const [stats, list] = await Promise.all([GrammarService.dashboard(), GrammarService.listUnits()]);
    setDashboard(stats);
    setUnits(list);
    if (!selectedId && list[0]) setSelectedId(list[0].id);
  }, [selectedId]);

  useEffect(() => {
    load().catch(() => undefined);
  }, [load]);

  useEffect(() => {
    if (focusUnitId) setSelectedId(focusUnitId);
  }, [focusUnitId]);

  useEffect(() => {
    if (!selected) return;
    setResult(null);
    setAnswers({});
    setChecked({});
    Promise.all([
      GrammarService.listExercises(selected.id),
      GrammarService.linkedCards(selected.id),
    ]).then(([items, cards]) => {
      setExercises(items);
      setPractice(items.slice(0, 8));
      setLinkedCards(cards);
    }).catch(() => undefined);
  }, [selected?.id]);

  async function createUnit() {
    if (!draft.title.trim()) return;
    setBusy(true);
    try {
      const created = await GrammarService.createUnit(draft);
      await Promise.all(
        demoExercises.map((ex) => GrammarService.saveExercise({ ...ex, unitId: created.id }))
      );
      setDraft(emptyUnit);
      setSelectedId(created.id);
      await load();
    } finally {
      setBusy(false);
    }
  }

  async function startPractice(control = false) {
    if (!selected) return;
    const items = await GrammarService.practice(selected.id, control ? 3 : 8);
    setPractice(items);
    setAnswers({});
    setChecked({});
    setResult(null);
  }

  async function finishPractice() {
    if (!selected || practice.length === 0) return;
    const correct = practice.filter(
      (ex) => normalize(answers[ex.id] ?? "") === normalize(ex.answer)
    ).length;
    const next = await GrammarService.completeSession(selected.id, correct, practice.length);
    setResult(next);
    await load();
  }

  function exportSelected() {
    if (!selected) return;
    const payload = {
      version: 1,
      unit: {
        title: selected.title, language: selected.language, level: selected.level,
        section: selected.section, tags: selected.tags, theory: selected.theory,
        examples: selected.examples,
      },
      exercises: exercises.map(({ type, prompt, payload, answer, explanation }) => ({
        type, prompt, payload, answer, explanation,
      })),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${selected.title.replace(/[^\wа-яё-]+/gi, "_") || "grammar-unit"}.linguadeck-grammar.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  async function importUnit(file: File | null) {
    if (!file) return;
    const raw = await file.text();
    const parsed = JSON.parse(raw) as {
      unit?: GrammarUnitInput;
      exercises?: Array<{ type: GrammarExerciseType; prompt: string; payload?: string; answer: string; explanation?: string }>;
    };
    if (!parsed.unit?.title) return;
    const created = await GrammarService.createUnit(parsed.unit);
    await Promise.all(
      (parsed.exercises ?? []).map((ex) =>
        GrammarService.saveExercise({
          unitId: created.id, type: ex.type, prompt: ex.prompt,
          payload: ex.payload ?? "{}", answer: ex.answer, explanation: ex.explanation ?? "",
        })
      )
    );
    setSelectedId(created.id);
    await load();
  }

  const importDefaultModules = useCallback(async () => {
    await GrammarService.importModules(englishGrammarModules);
    await load();
  }, [load]);

  return {
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
  };
}
