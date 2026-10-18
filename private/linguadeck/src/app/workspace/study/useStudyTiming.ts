import { useRef } from "react";
import { ViewService } from "../../../services/db/ViewService";
import type { Card } from "../../../types/domain";

export interface StudyTimingRefs {
  faceStartRef: React.MutableRefObject<number>;
  cardStartRef: React.MutableRefObject<number>;
  frontTimeRef: React.MutableRefObject<number>;
  hasFlippedRef: React.MutableRefObject<boolean>;
  flipsCountRef: React.MutableRefObject<number>;
  backtracksCountRef: React.MutableRefObject<number>;
}

export function useStudyTiming() {
  const faceStartRef = useRef<number>(Date.now());
  const cardStartRef = useRef<number>(Date.now());
  const frontTimeRef = useRef<number>(0);
  const hasFlippedRef = useRef<boolean>(false);
  const flipsCountRef = useRef<number>(0);
  const backtracksCountRef = useRef<number>(0);

  function resetCardTiming() {
    const now = Date.now();
    faceStartRef.current = now;
    cardStartRef.current = now;
    frontTimeRef.current = 0;
    hasFlippedRef.current = false;
    flipsCountRef.current = 0;
    backtracksCountRef.current = 0;
  }

  function elapsedMs() {
    return Date.now() - faceStartRef.current;
  }

  /**
   * Вызывается при перевороте карточки (лицо → оборот).
   * Фиксирует время, проведённое на лицевой стороне, и сбрасывает таймер грани.
   * FIX: ранее frontTimeRef никогда не обновлялся — collectMetrics() всегда
   * возвращал frontTimeMs: 0, из-за чего дашборд показывал 0 время учёбы.
   */
  function recordFrontTime() {
    const ms = Date.now() - faceStartRef.current;
    frontTimeRef.current = ms;
    faceStartRef.current = Date.now();
    hasFlippedRef.current = true;
    flipsCountRef.current += 1;
  }

  function logView(card: Card, type: "view" | "view_flipped", timeMs?: number) {
    console.log(`[timing] logView card=${card.id} type=${type} timeMs=${timeMs}`);
    ViewService.logView(card.id, card.deck_id, type, timeMs);
  }

  function collectMetrics() {
    return {
      frontTimeMs: frontTimeRef.current,
      totalTimeMs: Date.now() - cardStartRef.current,
      flipsCount: flipsCountRef.current,
      backtracksCount: backtracksCountRef.current,
    };
  }

  return {
    faceStartRef,
    cardStartRef,
    frontTimeRef,
    hasFlippedRef,
    flipsCountRef,
    backtracksCountRef,
    resetCardTiming,
    elapsedMs,
    recordFrontTime,
    logView,
    collectMetrics,
  };
}
