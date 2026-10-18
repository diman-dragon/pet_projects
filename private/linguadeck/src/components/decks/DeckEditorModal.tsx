import type { Deck } from "../../types/domain";
import { Button, Modal, Input, Select, TextArea } from "../../ui";
import type { DeckDraft } from "../../app/workspace/useDecksController";

interface DeckEditorModalProps {
  open: boolean;
  value: DeckDraft;
  editingDeck: Deck | null;
  onChange: (patch: Partial<DeckDraft>) => void;
  onSave: () => void;
  onClose: () => void;
}

import { LANGUAGE_OPTIONS } from "../../app/languageConfig";

export function DeckEditorModal({
  open,
  value,
  editingDeck,
  onChange,
  onSave,
  onClose,
}: DeckEditorModalProps) {
  const isCustom = value.language !== "" && !LANGUAGE_OPTIONS.some((o) => o.value === value.language);
  const selectValue = isCustom ? "other" : value.language;

  function handleSelectChange(val: string) {
    if (val === "other") {
      onChange({ language: "" });
    } else {
      onChange({ language: val });
    }
  }

  return (
    <Modal
      isOpen={open}
      title={editingDeck ? "Редактирование колоды" : "Новая колода"}
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose} variant="ghost">Отмена</Button>
          <Button onClick={onSave} variant="primary">
            Сохранить
          </Button>
        </>
      }
    >
      <div className="space-y-6">
        <Input
          label="Название"
          placeholder="Например: Английские фразовые глаголы"
          value={value.name}
          onChange={(e) => onChange({ name: e.target.value })}
        />

        <div className="space-y-3">
          <Select
            label="Язык колоды"
            options={LANGUAGE_OPTIONS.map(o => ({ value: o.value, label: `${o.flag ? o.flag + " " : ""}${o.label}` }))}
            value={selectValue}
            onChange={handleSelectChange}
          />
          
          {selectValue === "other" && (
            <Input
              placeholder="Введите название языка"
              value={value.language}
              onChange={(e) => onChange({ language: e.target.value })}
            />
          )}
        </div>

        <TextArea
          label="Описание"
          placeholder="Кратко опишите тематику колоды"
          value={value.description}
          onChange={(e) => onChange({ description: e.target.value })}
        />
      </div>
    </Modal>
  );
}
