import { useState, useCallback } from "react";

export interface CardBehavior {
  frontTimeMs: number;
  totalTimeMs: number;
  flipsCount: number;
  backtracksCount: number;
  manualWrong: boolean;
}

export function useBehaviorCollector() {
  const [startTime, setStartTime] = useState<number>(0);
  const [revealTime, setRevealTime] = useState<number>(0);
  const [flips, setFlips] = useState(0);
  const [manualWrong, setManualWrong] = useState(false);
  
  const startSession = useCallback(() => {
    setStartTime(performance.now());
    setRevealTime(0);
    setFlips(0);
    setManualWrong(false);
  }, []);

  const recordReveal = useCallback(() => {
    if (revealTime === 0) {
      setRevealTime(performance.now());
    }
    setFlips(prev => prev + 1);
  }, [revealTime]);

  const recordManualWrong = useCallback(() => {
    setManualWrong(true);
  }, []);

  const getMetrics = useCallback((): CardBehavior => {
    const end = performance.now();
    return {
      frontTimeMs: Math.round(revealTime > 0 ? revealTime - startTime : end - startTime),
      totalTimeMs: Math.round(end - startTime),
      flipsCount: flips,
      backtracksCount: 0,
      manualWrong,
    };
  }, [startTime, revealTime, flips, manualWrong]);

  return {
    startSession,
    recordReveal,
    recordManualWrong,
    getMetrics,
    manualWrong,
  };
}
