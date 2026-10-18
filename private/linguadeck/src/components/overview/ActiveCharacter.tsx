export function ActiveCharacter() {
  return (
    <div className="relative h-24 w-24 mx-auto mb-4">
      <div className="absolute inset-0 rounded-full bg-gradient-to-tr from-blue-400 to-indigo-500 animate-[bounce_2s_infinite]"></div>
      <div className="absolute inset-2 rounded-full bg-white opacity-20 animate-pulse"></div>
      <div className="absolute top-4 left-6 h-3 w-3 rounded-full bg-white animate-[bounce_1s_infinite]"></div>
      <div className="absolute top-4 right-6 h-3 w-3 rounded-full bg-white animate-[bounce_1s_infinite]"></div>
      <div className="absolute bottom-6 left-6 h-2 w-12 rounded-full bg-white/40 animate-pulse"></div>
    </div>
  );
}
