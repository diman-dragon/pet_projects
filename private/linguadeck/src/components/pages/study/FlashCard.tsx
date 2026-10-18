import { memo } from "react";
import { motion } from "framer-motion";
import type { Card } from "../../../types/domain";
import { CardFront } from "./CardFront";
import { CardBack } from "./CardBack";

interface FlashCardProps {
  card: Card;
  rating?: number;
  flipped: boolean;
  flipCount: number;
  onFlip: () => void;
  audioAvailable: boolean;
  videoAvailable: boolean;
  audioPlaying: boolean;
  onPlayAudio: () => void;
  onPlayVideo: () => void;
}

export const FlashCard = memo(function FlashCard({
  card, rating, flipped, flipCount, onFlip,
  audioAvailable, videoAvailable, audioPlaying, onPlayAudio, onPlayVideo,
}: FlashCardProps) {
  return (
    <div className="relative w-full h-full min-h-[400px] flex items-center justify-center">
      <motion.div
        className="relative w-full h-full max-h-[600px] aspect-[4/3] sm:aspect-[16/10] lg:aspect-[16/9] cursor-pointer select-none"
        style={{ perspective: "1600px" }}
      >
        <motion.div
          key={card.id}
          initial={{ opacity: 0, scale: 0.94, y: 20 }}
          animate={{ 
            opacity: 1, 
            scale: 1, 
            y: 0, 
            rotateY: flipped ? 180 : 0 
          }}
          exit={{ opacity: 0, scale: 1.05, y: -20 }}
          transition={{
            opacity: { duration: 0.25 },
            rotateY: { duration: 0.6, ease: [0.23, 1, 0.32, 1] },
            scale: { duration: 0.4 },
            y: { duration: 0.4 }
          }}
          className="relative w-full h-full"
          style={{ transformStyle: "preserve-3d" }}
          onClick={onFlip}
        >
          <div
            key={`shimmer-${flipCount}`}
            className="pointer-events-none absolute inset-0 z-20 overflow-hidden rounded-[40px]"
            style={{ backfaceVisibility: "hidden" }}
          >
            <motion.div
              className="absolute inset-y-0 w-2/5 bg-gradient-to-r from-transparent via-white/40 to-transparent"
              initial={{ x: "-100%" }}
              animate={{ x: "400%" }}
              transition={{ duration: 0.8, ease: "easeInOut" }}
            />
          </div>

          <CardFront
            card={card}
            rating={rating}
            audioAvailable={audioAvailable}
            videoAvailable={videoAvailable}
            audioPlaying={audioPlaying}
            onPlayAudio={onPlayAudio}
            onPlayVideo={onPlayVideo}
          />
          <CardBack card={card} rating={rating} flipCount={flipCount} />
        </motion.div>
      </motion.div>
    </div>
  );
});
