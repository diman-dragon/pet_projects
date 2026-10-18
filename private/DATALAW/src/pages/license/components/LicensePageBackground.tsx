// src/pages/license/components/LicensePageBackground.tsx
//
// Фоновый слой страницы активации.
// Сетка + горизонтальные сканирующие линии + угловые маркеры.
// Стилистика: оперативный центр, защищённая система.

export function LicensePageBackground() {
  return (
    <div className="absolute inset-0 z-0 overflow-hidden" aria-hidden>
      {/* Базовый градиент — холодный серо-синий */}
      <div
        className="absolute inset-0"
        style={{
          background: 'linear-gradient(160deg, #f0f2f5 0%, #e8edf4 40%, #eef0f5 100%)',
        }}
      />

      {/* Тонкая сетка */}
      <div
        className="absolute inset-0 opacity-[0.04]"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0,103,192,1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,103,192,1) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px',
        }}
      />

      {/* Горизонтальная сканирующая линия */}
      <div
        className="absolute left-0 right-0 h-[2px] opacity-20"
        style={{
          background: 'linear-gradient(90deg, transparent, #0067c0 20%, #0067c0 80%, transparent)',
          animation: 'scanline 6s ease-in-out infinite',
          top: '30%',
        }}
      />

      {/* Угловые маркеры — как в оперативных системах */}
      {[
        'top-6 left-6 border-t-2 border-l-2',
        'top-6 right-6 border-t-2 border-r-2',
        'bottom-6 left-6 border-b-2 border-l-2',
        'bottom-6 right-6 border-b-2 border-r-2',
      ].map((cls, i) => (
        <div key={i} className={`absolute w-8 h-8 border-black/10 ${cls}`} />
      ))}

      {/* Диагональные декоративные линии */}
      <div
        className="absolute top-0 left-0 w-full h-full opacity-[0.015]"
        style={{
          backgroundImage:
            'repeating-linear-gradient(45deg, #0067c0 0, #0067c0 1px, transparent 0, transparent 50%)',
          backgroundSize: '20px 20px',
        }}
      />

      <style>{`
        @keyframes scanline {
          0%   { top: 15%; opacity: 0; }
          10%  { opacity: 0.2; }
          45%  { opacity: 0.15; }
          90%  { opacity: 0.2; }
          100% { top: 85%; opacity: 0; }
        }
      `}</style>
    </div>
  );
}
