// components/orb/LailaOrb.tsx

import { OrbSize } from '../../types';
import { OrbPulse } from './OrbPulse';
import { OrbCore } from './OrbCore';
import { OrbOrbitSystem } from './OrbOrbitSystem';

interface LailaOrbProps {
  size?: OrbSize;
  isActive?: boolean;
}

const CONTAINER_SIZES: Record<OrbSize, string> = {
  sm: 'size-6',
  md: 'size-10',
  lg: 'size-16',
  fab: 'size-12',
};

const PLANET_SIZES: Record<OrbSize, string> = {
  sm: 'size-1',
  md: 'size-1.5',
  lg: 'size-2.5',
  fab: 'size-2',
};

/**
 * Главный компонент-оболочка орба Laila.
 * Собирает: пульс → ядро → орбитальная система.
 */
export function LailaOrb({ size = 'md', isActive = false }: LailaOrbProps) {
  return (
    <div className={`${CONTAINER_SIZES[size]} relative flex items-center justify-center shrink-0`}>
      {isActive && <OrbPulse />}
      <OrbCore isActive={isActive} />
      <OrbOrbitSystem planetSize={PLANET_SIZES[size]} />
    </div>
  );
}
