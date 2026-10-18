import { useCallback, useEffect, useState, useMemo } from "react";
import type { Card, ReviewRating, WeekActivityDay } from "../../../types/domain";
import { CardStatsService, type CardStats } from "../../../services/db/CardStatsService";
import { GrammarService } from "../../../services/db/GrammarService";
import { useBehaviorCollector } from "../../../hooks/useBehaviorCollector";

interface UseStudyPageProps {
  card: Card | null;
  flipped: boolean;
  inferredRating: ReviewRating | null;
  reviewedToday?: number;
  weekActivity?: WeekActivityDay[];
  onFlip: () => void;
  onComplete: (behavior?: any) => Promise<boolean>;
  onPlayAudio?: (audioPath: string) => void | Promise<void>;
}

const WEEK_LABELS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];

export function useStudyPage({
  card,
  reviewedToday = 0,
  weekActivity,
  onFlip,
  onComplete,
  onPlayAudio,
}: UseStudyPageProps) {
  const [statsOpen, setStatsOpen] = useState(false);
  const [stats, setStats] = useState<CardStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [flipCount, setFlipCount] = useState(0);
  const [streakOpen, setStreakOpen] = useState(false);
  const [hadCard, setHadCard] = useState(false);
  const [audioPlaying, setAudioPlaying] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [grammarLinks, setGrammarLinks] = useState<Array<{ id: number; title: string }>>([]);

  const behavior = useBehaviorCollector();

  useEffect(() => {
    setAudioPlaying(false);
    setAudioError(null);
    if (card) {
      behavior.startSession();
    }
  }, [card?.id]);

  const handlePlayAudio = useCallback(async () => {
    if (!card?.audio_path || !onPlayAudio || audioPlaying) return;
    setAudioPlaying(true);
    setAudioError(null);
    try {
      await onPlayAudio(card.audio_path);
    } catch (err) {
      setAudioError(err instanceof Error ? err.message : "Ошибка воспроизведения");
    } finally {
      setAudioPlaying(false);
    }
  }, [card?.audio_path, onPlayAudio, audioPlaying]);

  const handleComplete = useCallback(() => {
    setAudioPlaying(false);
    onComplete(behavior.getMetrics());
  }, [onComplete, behavior]);

  const handleNext = useCallback(() => {
    handleComplete();
  }, [handleComplete]);

  const handleFlip = useCallback(() => {
    setFlipCount((c) => c + 1);
    behavior.recordReveal();
    onFlip();
  }, [onFlip, behavior]);

  const handleManualWrong = useCallback(() => {
    behavior.recordManualWrong();
    setTimeout(() => {
      onComplete({ ...behavior.getMetrics(), manualWrong: true });
    }, 10);
  }, [behavior, onComplete]);

  useEffect(() => {
    if (card) setHadCard(true);
  }, [card]);

  useEffect(() => {
    if (hadCard && !card) {
      setStreakOpen(true);
      setHadCard(false);
    }
  }, [card, hadCard]);

  useEffect(() => {
    setFlipCount(0);
  }, [card?.id]);

  useEffect(() => {
    if (!card) {
      setGrammarLinks([]);
      return;
    }
    GrammarService.unitsForCard(card.id)
      .then((units) => setGrammarLinks(units.map((unit) => ({ id: unit.id, title: unit.title }))))
      .catch(() => setGrammarLinks([]));
  }, [card?.id]);

  useEffect(() => {
    if (!card) return;
    setStats(null);
    setStatsLoading(true);
    CardStatsService.get(card.id)
      .then(setStats)
      .catch(() => setStats(null))
      .finally(() => setStatsLoading(false));
  }, [card?.id]);

  const today = new Date().getDay();
  const todayMon = today === 0 ? 6 : today - 1;
  const weekBars = useMemo(() => {
    if (weekActivity?.length) return weekActivity;
    return WEEK_LABELS.map((label, i) => ({
      date: "",
      label,
      weekday: i,
      timeMs: 0,
      count: i === todayMon ? reviewedToday : 0,
      isToday: i === todayMon,
      hasActivity: i === todayMon && reviewedToday > 0,
    }));
  }, [weekActivity, todayMon, reviewedToday]);

  return {
    statsOpen,
    setStatsOpen,
    stats,
    statsLoading,
    flipCount,
    streakOpen,
    setStreakOpen,
    audioPlaying,
    audioError,
    setAudioError,
    grammarLinks,
    handlePlayAudio,
    handleFlip,
    handleNext,
    handleManualWrong,
    weekBars,
  };
}
