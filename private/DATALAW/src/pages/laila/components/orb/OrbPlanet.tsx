// components/orb/OrbPlanet.tsx

interface OrbPlanetProps {
  size: string; // Tailwind size class, e.g. 'size-2'
  position: React.CSSProperties;
  rotateTransform?: string;
}

const PLANET_SHADOW = `0 0 8px rgba(233,196,106,0.8), inset -1px -1px 2px rgba(0,0,0,0.4)`;

/** Один золотой шарик-спутник на орбите */
export function OrbPlanet({ size, position, rotateTransform }: OrbPlanetProps) {
  return (
    <div
      className={`${size} absolute bg-[#E9C46A] rounded-full`}
      style={{
        ...position,
        boxShadow: PLANET_SHADOW,
        transform: rotateTransform,
      }}
    />
  );
}
