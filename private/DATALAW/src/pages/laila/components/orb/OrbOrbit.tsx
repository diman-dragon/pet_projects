// components/orb/OrbOrbit.tsx

import { OrbPlanet } from './OrbPlanet';

interface OrbOrbitProps {
  animationClass: string; // e.g. 'animate-orb-x'
  ringOpacity: string; // e.g. 'border-[#E9C46A]/30'
  planetSize: string; // e.g. 'size-2'
  planetPosition: React.CSSProperties;
  planetRotate?: string;
}

/** Одна орбита: кольцо + спутник */
export function OrbOrbit({
  animationClass,
  ringOpacity,
  planetSize,
  planetPosition,
  planetRotate,
}: OrbOrbitProps) {
  return (
    <div className={`absolute inset-0 ${animationClass}`} style={{ transformStyle: 'preserve-3d' }}>
      {/* Кольцо орбиты */}
      <div className={`absolute inset-0 rounded-full border ${ringOpacity}`} />

      {/* Спутник */}
      <OrbPlanet size={planetSize} position={planetPosition} rotateTransform={planetRotate} />
    </div>
  );
}
