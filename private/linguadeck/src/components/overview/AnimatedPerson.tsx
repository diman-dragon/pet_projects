export function Person({ color }: { color: string }) {
  return (
    <g>
      {/* Body Sway Animation Wrapper */}
      <g>
        <animateTransform 
            attributeName="transform" type="rotate" 
            values="-5 0 -15; 5 0 -15; -5 0 -15" 
            dur="2s" repeatCount="indefinite" 
        />
        {/* Head */}
        <circle cx="0" cy="-32" r="7" fill="#fecaca" stroke="#475569" strokeWidth="1.5" />
        {/* Hair - Stylized */}
        <path d="M-6 -38 C -8 -45, 8 -45, 6 -38" fill="#1e293b" />
        {/* Eyes */}
        <circle cx="-2.5" cy="-34" r="0.8" fill="#1e293b" />
        <circle cx="2.5" cy="-34" r="0.8" fill="#1e293b" />
        {/* Mouth */}
        <path d="M-2 -29 Q 0 -27 2 -29" stroke="#9f1239" fill="none" strokeWidth="1" strokeLinecap="round" />
        
        {/* Torso */}
        <path d="M-7 -25 L 7 -25 L 6 -10 L -6 -10 Z" fill={color} stroke="#475569" strokeWidth="1.5" />
      </g>

      {/* Arms (Independent Sway) */}
      <path d="M-8 -22 L -14 -10" stroke={color} strokeWidth="3" strokeLinecap="round">
        <animateTransform attributeName="transform" type="rotate" values="-15 -8 -22; 15 -8 -22; -15 -8 -22" dur="1s" repeatCount="indefinite" />
      </path>
      <path d="M8 -22 L 14 -10" stroke={color} strokeWidth="3" strokeLinecap="round">
        <animateTransform attributeName="transform" type="rotate" values="15 8 -22; -15 8 -22; 15 8 -22" dur="1s" repeatCount="indefinite" />
      </path>

      {/* Legs */}
      <path d="M-4 -10 L -4 2" stroke="#475569" strokeWidth="3" strokeLinecap="round" />
      <path d="M4 -10 L 4 2" stroke="#475569" strokeWidth="3" strokeLinecap="round" />
    </g>
  );
}
