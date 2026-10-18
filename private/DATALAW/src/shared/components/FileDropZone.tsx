// src/shared/components/FileDropZone.tsx
import { useState, useRef, ReactNode } from 'react';
import { Upload, Loader2 } from 'lucide-react';
import { UiText } from '../../components/ui/atoms/UiText';

interface FileDropZoneProps {
  onFilesDrop: (files: File[]) => void;
  children: ReactNode;
  className?: string;
  accept?: string;
  isUploading?: boolean;
}

export function FileDropZone({
  onFilesDrop,
  children,
  className = '',
  accept,
  isUploading = false,
}: FileDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isUploading) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (isUploading) return;

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      onFilesDrop(files);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      onFilesDrop(files);
    }
  };

  return (
    <div
      className={`relative group ${className}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        multiple
        accept={accept}
        onChange={handleFileInputChange}
      />

      {children}

      {/* Overlay for Drag & Drop */}
      {(isDragging || isUploading) && (
        <div className={`absolute inset-0 z-50 flex flex-col items-center justify-center gap-3 backdrop-blur-md transition-all duration-300 ${isDragging ? 'bg-[var(--app-accent)]/10 border-2 border-dashed border-[var(--app-accent)]/40' : 'bg-white/60'
          }`}>
          <div className={`w-16 h-16 rounded-full flex items-center justify-center shadow-xl ${isDragging ? 'bg-[var(--app-accent)] text-white animate-bounce' : 'bg-white text-[var(--app-accent)]'
            }`}>
            {isUploading ? (
              <Loader2 size={32} className="animate-spin" />
            ) : (
              <Upload size={32} />
            )}
          </div>

          <div className="flex flex-col items-center gap-1 bg-white/90 px-6 py-3 rounded-2xl shadow-sm border border-black/[0.05]">
            <UiText size="sm" weight="black" uppercase tracking="tight">
              {isUploading ? 'Загрузка материалов...' : 'Перетащите файлы сюда'}
            </UiText>
            {!isUploading && (
              <UiText size="xs" tone="muted">или нажмите для выбора</UiText>
            )}
          </div>

          {!isUploading && (
            <button
              className="absolute inset-0 w-full h-full cursor-pointer opacity-0"
              onClick={() => fileInputRef.current?.click()}
            />
          )}
        </div>
      )}

      {/* Click-to-upload area (removed to keep only one button in header) */}
    </div>
  );
}
