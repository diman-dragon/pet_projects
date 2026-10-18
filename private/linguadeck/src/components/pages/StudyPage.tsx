import type { Card } from "../../types/domain";
import { StudyHeader } from "./study/StudyHeader";
import { StudyControls } from "./study/StudyControls";
import { SessionSummaryPanel } from "../study/SessionSummaryPanel";
import { GrammarLinks } from "./study/GrammarLinks";
import { NoActiveSession } from "./study/NoActiveSession";
import { useStudyHotkeys } from "../../app/workspace/useStudyHotkeys";
import { useStudy } from "../../app/workspace/study/StudyContext";
import { useStudyPage } from "./hooks/useStudyPage";
import { CardDisplayArea } from "./study/containers/CardDisplayArea";
import { StudyFeedbackArea } from "./study/containers/StudyFeedbackArea";

export interface StudyPageProps {
  card: Card | null;
  deckName: string;
  total: number;
  streak?: number;
  sessionCount?: number;
  reviewedToday?: number;
  totalTimeMs?: number;
  weekActivity?: any[];
  onPlayAudio?: (audioPath: string) => void | Promise<void>;
  onPlayVideo?: (videoPath: string) => void;
  onEditCard?: (card: Card) => void;
  onOpenGrammarTopic?: (unitId: number) => void;
}

export function StudyPage(props: StudyPageProps) {
  const study = useStudy();
  const card = study.currentStudyCard;

  const {
    statsOpen, setStatsOpen, stats, statsLoading,
    audioPlaying, audioError, setAudioError, grammarLinks,
    handlePlayAudio, handleFlip, handleNext, handleManualWrong,
  } = useStudyPage({
    card, 
    flipped: study.studyFlipped, 
    inferredRating: null,
    reviewedToday: props.reviewedToday,
    weekActivity: props.weekActivity,
    onFlip: study.flipStudyCard, 
    onComplete: study.goToNextStudyCard,
    onPlayAudio: props.onPlayAudio,
  });

  useStudyHotkeys({
    enabled: !!card && !statsOpen,
    flipped: study.studyFlipped,
    total: study.studyCards.length,
    onFlip: handleFlip,
    onNext: handleNext,
    onPrev: study.goToPreviousStudyCard,
    onManualWrong: handleManualWrong,
  });

  if (study.sessionSummary !== null) {
    return (
      <SessionSummaryPanel
        summary={study.sessionSummary}
        deckName={props.deckName}
        onContinue={study.clearSession}
      />
    );
  }

  if (!card) {
    return <NoActiveSession />;
  }

  return (
    <div className="flex h-full flex-col overflow-hidden bg-gradient-to-br from-slate-50 via-blue-50/20 to-slate-100">
      <StudyHeader
        deckName={props.deckName} index={study.studyIndex} total={study.studyCards.length}
        statsOpen={statsOpen} onToggleStats={() => setStatsOpen(v => !v)}
        onEdit={props.onEditCard && card ? () => props.onEditCard!(card) : undefined}
        onFinish={study.finishSession}
      />

      <StudyFeedbackArea 
        isSaving={study.isSaving} 
        audioError={audioError} 
        onClearAudioError={() => setAudioError(null)} 
      />

      <CardDisplayArea
        statsOpen={statsOpen}
        card={card}
        stats={stats}
        statsLoading={statsLoading}
        flipped={study.studyFlipped}
        audioPlaying={audioPlaying}
        audioAvailable={Boolean(card.audio_path && props.onPlayAudio)}
        videoAvailable={Boolean(card.video_path && props.onPlayVideo)}
        onCloseStats={() => setStatsOpen(false)}
        onFlip={handleFlip}
        onPlayAudio={handlePlayAudio}
        onPlayVideo={() => props.onPlayVideo?.(card.video_path!)}
      />

      <GrammarLinks links={grammarLinks} onOpen={(id) => props.onOpenGrammarTopic?.(id)} />

      <StudyControls 
        onFlip={handleFlip}
        onPrev={study.goToPreviousStudyCard}
        onNext={handleNext}
      />
    </div>
  );
}
