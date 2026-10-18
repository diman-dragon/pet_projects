// components/orb/OrbPulse.tsx

/** Внешний пульсирующий ореол (только когда isActive) */
export function OrbPulse() {
  return (
    <div
      className="absolute inset-0 rounded-full bg-[#E9C46A]/20 animate-ping z-0"
      style={{ animationDuration: '3s' }}
    />
  );
}
