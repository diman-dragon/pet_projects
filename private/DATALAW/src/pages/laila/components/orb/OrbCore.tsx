// components/orb/OrbCore.tsx

interface OrbCoreProps {
  isActive: boolean;
}

const COLOR_GOLD = '#E9C46A';

/** Центральное золотое ядро с 3D-свечением */
export function OrbCore({ isActive }: OrbCoreProps) {
  return (
    <div
      className="absolute rounded-full transition-all duration-700 z-10"
      style={{
        width: '40%',
        height: '40%',
        background: COLOR_GOLD,
        boxShadow: isActive
          ? `inset -2px -2px 5px rgba(0,0,0,0.3), 0 0 25px rgba(233,196,106,0.6)`
          : `inset -1px -1px 3px rgba(0,0,0,0.2), 0 0 10px rgba(233,196,106,0.2)`,
      }}
    />
  );
}
