import { AnimatePresence } from "framer-motion";
import { CardStatsPanel } from "../CardStatsPanel";
import { FlashCard } from "../FlashCard";
import type { Card } from "../../../../types/domain";

interface CardDisplayAreaProps {
  statsOpen: boolean;
  card: Card;
  stats: any;
  statsLoading: boolean;
  flipped: boolean;
  audioPlaying: boolean;
  audioAvailable: boolean;
  videoAvailable: boolean;
  onCloseStats: () => void;
  onFlip: () => void;
  onPlayAudio: () => void;
  onPlayVideo: () => void;
}

export function CardDisplayArea({
  statsOpen, card, stats, statsLoading, flipped, 
  audioPlaying, audioAvailable, videoAvailable,
  onCloseStats, onFlip, onPlayAudio, onPlayVideo
}: CardDisplayAreaProps) {
  return (
    <div className="flex flex-1 min-h-0 px-6 py-4 relative group">
      <AnimatePresence mode="wait">
        {statsOpen ? (
          <CardStatsPanel key="stats" card={card} stats={stats} loading={statsLoading} onClose={onCloseStats} />
        ) : (
          <div className="w-full flex flex-col max-w-4xl mx-auto">
            <div className="flex-1 relative min-h-0">
              <FlashCard
                key={card.id}
                card={card} 
                rating={stats?.recent_reviews?.[0]?.rating}
                flipped={flipped} 
                flipCount={0}
                onFlip={onFlip}
                audioAvailable={audioAvailable}
                videoAvailable={videoAvailable}
                audioPlaying={audioPlaying}
                onPlayAudio={onPlayAudio}
                onPlayVideo={onPlayVideo}
              />
            </div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
