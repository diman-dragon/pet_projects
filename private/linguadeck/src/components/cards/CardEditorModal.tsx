import { AlertCircle, ImagePlus, PlaySquare, Save, Volume2 } from "lucide-react";
import { useMemo } from "react";
import type { CreateCardInput } from "../../services/db/CardService";
import { Button, Modal, Input, TextArea } from "../../ui";

interface CardEditorModalProps {
  open: boolean;
  editing: boolean;
  draft: CreateCardInput;
  deckName: string | null;
  saving?: boolean;
  externalError?: string;
  onChange: (field: keyof CreateCardInput, value: string) => void;
  onPickAudio: () => void;
  onPickVideo: () => void;
  onPickImage: () => void;
  onSave: () => void;
  onClose: () => void;
}

const mainFields: Array<{ key: keyof CreateCardInput; label: string; placeholder?: string }> = [
  { key: "front", label: "Слово", placeholder: "Например: enthusiasm" },
  { key: "back", label: "Перевод", placeholder: "Например: энтузиазм" },
  { key: "transcription", label: "Транскрипция", placeholder: "[ɪnˈθuːziæzəm]" },
];

const contextFields: Array<{ key: keyof CreateCardInput; label: string; multiline?: boolean }> = [
  { key: "context", label: "Контекст", multiline: true },
  { key: "etymology", label: "Этимология", multiline: true },
  { key: "mnemonic", label: "Мнемоника", multiline: true },
  { key: "example_1", label: "Пример 1", multiline: true },
  { key: "example_2", label: "Пример 2", multiline: true },
  { key: "example_3", label: "Пример 3", multiline: true },
  { key: "notes", label: "Заметки", multiline: true },
  { key: "tags", label: "Теги" },
];

export function CardEditorModal({
  open,
  editing,
  draft,
  deckName,
  saving = false,
  externalError,
  onChange,
  onPickAudio,
  onPickVideo,
  onPickImage,
  onSave,
  onClose,
}: CardEditorModalProps) {
  const errors = useMemo(() => {
    const errs: string[] = [];
    if (!draft.front?.toString().trim()) errs.push("Слово (front)");
    if (!draft.back?.toString().trim()) errs.push("Перевод (back)");
    return errs;
  }, [draft.front, draft.back]);

  const hasErrors = errors.length > 0;
  const isDisabled = hasErrors || saving;

  return (
    <Modal
      isOpen={open}
      title={editing ? "Редактирование карточки" : "Новая карточка"}
      onClose={onClose}
      size="lg"
      footer={
        <>
          <Button onClick={onClose} disabled={saving} variant="ghost">
            Отмена
          </Button>
          <Button
            onClick={onSave}
            variant="primary"
            disabled={isDisabled}
            isLoading={saving}
            leftIcon={<Save size={16} />}
          >
            Сохранить
          </Button>
        </>
      }
    >
      <div className="space-y-6 max-h-[60vh] overflow-y-auto pr-4 custom-scrollbar">
        {deckName && (
          <div className="px-4 py-2 bg-slate-50 border border-slate-100 rounded-2xl text-[10px] font-bold uppercase tracking-widest text-slate-400 w-fit">
            Колода: {deckName}
          </div>
        )}

        {/* Validation Errors */}
        {(hasErrors || externalError) && (
          <div className="rounded-[24px] bg-rose-50 border border-rose-100 p-5 flex items-start gap-3 animate-in fade-in slide-in-from-top-2">
            <AlertCircle size={20} className="text-rose-500 mt-0.5 flex-shrink-0" />
            <div className="text-sm font-medium text-rose-600">
              {hasErrors ? `Заполните обязательные поля: ${errors.join(", ")}` : externalError}
            </div>
          </div>
        )}

        {/* Main Fields */}
        <div className="grid gap-4 sm:grid-cols-2">
          {mainFields.map((field) => (
            <Input
              key={field.key}
              label={field.label + (field.key !== "transcription" ? " *" : "")}
              value={String(draft[field.key] ?? "")}
              onChange={(e) => onChange(field.key, e.target.value)}
              placeholder={field.placeholder}
              disabled={saving}
              containerClassName={field.key === "transcription" ? "sm:col-span-2" : ""}
            />
          ))}
        </div>

        {/* Media Buttons */}
        <div className="flex gap-3">
          <Button onClick={onPickAudio} className="flex-1" disabled={saving} variant="secondary" size="sm" leftIcon={<Volume2 size={16} />}>
            Аудио
          </Button>
          <Button onClick={onPickVideo} className="flex-1" disabled={saving} variant="secondary" size="sm" leftIcon={<PlaySquare size={16} />}>
            Видео
          </Button>
          <Button onClick={onPickImage} className="flex-1" disabled={saving} variant="secondary" size="sm" leftIcon={<ImagePlus size={16} />}>
            Фото
          </Button>
        </div>

        {/* Context Fields */}
        <div className="space-y-4">
          {contextFields.map((field) => (
            field.multiline ? (
              <TextArea
                key={field.key}
                label={field.label}
                value={String(draft[field.key] ?? "")}
                onChange={(e) => onChange(field.key, e.target.value)}
                disabled={saving}
                placeholder={`Введите ${field.label.toLowerCase()}...`}
              />
            ) : (
              <Input
                key={field.key}
                label={field.label}
                value={String(draft[field.key] ?? "")}
                onChange={(e) => onChange(field.key, e.target.value)}
                disabled={saving}
                placeholder={`Введите ${field.label.toLowerCase()}...`}
              />
            )
          ))}
        </div>
      </div>
    </Modal>
  );
}
