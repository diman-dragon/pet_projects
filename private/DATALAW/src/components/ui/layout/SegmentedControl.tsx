// src/components/ui/layout/SegmentedControl.tsx
import type { ReactNode } from 'react';
import { useRef, useEffect, useState } from 'react';

export interface Segment {
  id: string;
  label: string;
  icon?: ReactNode;
}

interface SegmentedControlProps {
  segments: Segment[];
  active: string;
  onChange: (id: string) => void;
  className?: string;
}

export function SegmentedControl({
  segments,
  active,
  onChange,
  className = '',
}: SegmentedControlProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [sliderStyle, setSliderStyle] = useState({ width: 0, left: 0 });

  useEffect(() => {
    if (!containerRef.current) return;

    const buttons = containerRef.current.querySelectorAll('button');
    const index = segments.findIndex((s) => s.id === active);
    const activeButton = buttons[index];

    if (activeButton) {
      const containerRect = containerRef.current.getBoundingClientRect();
      const buttonRect = activeButton.getBoundingClientRect();

      setSliderStyle({
        width: buttonRect.width - 4,
        left: buttonRect.left - containerRect.left + 2,
      });
    }
  }, [active, segments]);

  return (
    <div
      ref={containerRef}
      className={[
        'relative flex items-center gap-1 rounded-[8px] bg-black/[0.04] p-1 shadow-[inset_0_1px_2px_rgba(0,0,0,0.04)]',
        className,
      ].join(' ')}
    >
      <div
        className="absolute bottom-1 top-1 z-0 rounded-[6px] bg-white shadow-[0_1px_3px_rgba(0,0,0,0.1),0_1px_0_rgba(255,255,255,0.5)_inset] transition-all duration-200 ease-out"
        style={{
          width: sliderStyle.width,
          left: sliderStyle.left,
        }}
      />

      {segments.map((seg) => {
        const isActive = seg.id === active;

        return (
          <button
            key={seg.id}
            type="button"
            onClick={() => onChange(seg.id)}
            className={[
              'relative z-10 flex h-7 flex-1 items-center justify-center gap-1.5 whitespace-nowrap rounded-md px-3',
              'text-[length:var(--text-xs)] font-medium leading-[var(--leading-body)] transition-colors duration-200',
              isActive ? 'text-black' : 'text-black/50 hover:text-black',
            ].join(' ')}
          >
            {seg.icon && (
              <span className="text-[length:var(--text-sm)] leading-none">{seg.icon}</span>
            )}
            <span className="leading-none">{seg.label}</span>
          </button>
        );
      })}
    </div>
  );
}
