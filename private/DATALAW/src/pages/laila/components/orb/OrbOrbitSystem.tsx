// components/orb/OrbOrbitSystem.tsx

import { OrbOrbit } from './OrbOrbit';

interface OrbOrbitSystemProps {
  planetSize: string;
}

/**
 * Система трёх орбит с перспективной проекцией.
 * Каждая орбита отвечает за своё направление вращения.
 */
export function OrbOrbitSystem({ planetSize }: OrbOrbitSystemProps) {
  return (
    <div className="absolute inset-0 z-20" style={{ perspective: '300px' }}>
      {/* Орбита 1 — горизонтальная, быстрая */}
      <OrbOrbit
        animationClass="animate-orb-x"
        ringOpacity="border-[#E9C46A]/30"
        planetSize={planetSize}
        planetPosition={{ top: '50%', left: '-2%' }}
        planetRotate="rotateX(-70deg)"
      />

      {/* Орбита 2 — наклонная */}
      <OrbOrbit
        animationClass="animate-orb-y"
        ringOpacity="border-[#E9C46A]/20"
        planetSize={planetSize}
        planetPosition={{ bottom: '-2%', left: '50%' }}
        planetRotate="rotateX(30deg)"
      />

      {/* Орбита 3 — вертикальная, медленная */}
      <OrbOrbit
        animationClass="animate-orb-z"
        ringOpacity="border-[#E9C46A]/10"
        planetSize={planetSize}
        planetPosition={{ top: '-2%', left: '50%' }}
        planetRotate="rotateY(-90deg)"
      />
    </div>
  );
}
