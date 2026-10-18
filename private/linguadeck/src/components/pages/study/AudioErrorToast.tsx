interface AudioErrorToastProps {
  message: string;
  onClose: () => void;
}

export function AudioErrorToast({ message, onClose }: AudioErrorToastProps) {
  return (
    <div className="mx-5 mb-2 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
      <span>🔇</span>
      <span className="flex-1">{message}</span>
      <button className="text-red-400 hover:text-red-600" onClick={onClose}>✕</button>
    </div>
  );
}
