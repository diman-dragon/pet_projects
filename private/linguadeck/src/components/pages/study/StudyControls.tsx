import { Button } from "../../../ui";

interface StudyControlsProps {
  onFlip: () => void;
  onPrev: () => void;
  onNext: () => void;
}

export function StudyControls({ onFlip, onPrev, onNext }: StudyControlsProps) {
  return (
    <div className="flex items-center justify-center gap-4 px-5 pb-8">
      <Button 
        onClick={onPrev} 
        size="sm"
        variant="secondary"
      >
        Назад (←)
      </Button>

      <Button 
        onClick={onFlip} 
        size="lg"
        variant="primary"
      >
        <span>Перевернуть</span>
        <span className="rounded-md bg-white/20 px-1.5 py-0.5 text-[10px] font-black uppercase tracking-widest">Space</span>
      </Button>

      <Button 
        onClick={onNext} 
        size="sm"
        variant="secondary"
      >
        Вперед (→)
      </Button>
    </div>
  );
}
