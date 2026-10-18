// src/pages/LineDetails/scenarios/ConsultationQuestionScenario.tsx
import { Info, Plus, MessageSquare, Sparkles, Save } from 'lucide-react';
import { PageHeader } from '../../../components/ui/layout/PageHeader';
import { Window } from '../../../components/ui/layout/Window';
import { SegmentedControl } from '../../../components/ui/layout/SegmentedControl';
import { PageLoader } from '../../../components/ui/feedback/PageLoader';
import { Button } from '../../../components/ui/form/Button';
import { useI18n } from '../../../i18n/AppLanguageProvider';
import { AppealsTable } from '../components/AppealsTable';
import { InfoTab } from '../components/InfoTab';
import { AppealEditModal } from '../components/AppealEditModal';
import { DeleteConfirmModal } from '../../../components/ui/overlays/DeleteConfirmModal';
import { Appeal, Line } from '../../../types/entities';
import { Textarea } from '../../../components/ui/form/Textarea';
import { Select } from '../../../components/ui/form/Select';
import { useState, useEffect } from 'react';

interface ConsultationQuestionScenarioProps {
  line: Line;
  appeals: Appeal[];
  loading: boolean;
  view: 'appeals' | 'info';
  setView: (view: 'appeals' | 'info') => void;
  onBack: () => void;
  onNavigateAppeal: (appeal: Appeal) => void;
  // Modals logic
  isNew: boolean;
  editingAppeal: Appeal | null;
  isSaving: boolean;
  formData: any;
  setFormData: (data: any) => void;
  openCreate: () => void;
  openEdit: (appeal: Appeal) => void;
  closeModal: () => void;
  handleSaveAppeal: () => void;
  deleteTarget: Appeal | null;
  setDeleteTarget: (target: Appeal | null) => void;
  handleDeleteAppeal: () => void;
  handleUpdateLine: (updates: Partial<Line>) => Promise<void>;
}

export function ConsultationQuestionScenario({
  line,
  appeals,
  loading,
  view,
  setView,
  onBack,
  onNavigateAppeal,
  isNew,
  editingAppeal,
  isSaving,
  formData,
  setFormData,
  openCreate,
  openEdit,
  closeModal,
  handleSaveAppeal,
  deleteTarget,
  setDeleteTarget,
  handleDeleteAppeal,
  handleUpdateLine,
}: ConsultationQuestionScenarioProps) {
  const { t } = useI18n();
  const [localText, setLocalText] = useState(line.consultation_text || '');
  const [localTopic, setLocalTopic] = useState(line.consultation_topic || 'general');

  useEffect(() => {
    setLocalText(line.consultation_text || '');
    setLocalTopic(line.consultation_topic || 'general');
  }, [line]);

  const handleSaveText = () => {
    handleUpdateLine({
      consultation_text: localText,
      consultation_topic: localTopic,
    });
  };

  return (
    <div className="flex flex-col h-full bg-purple-50/30 overflow-hidden animate-in fade-in duration-300">
      <PageHeader
        title={t('scenario.consulting.topics')}
        subtitle={loading ? t('loader.loading') : line.title}
        icon={<MessageSquare size={20} className="text-purple-600" />}
        onBack={onBack}
        rightContent={
          <div className="flex items-center gap-3">
            <SegmentedControl
              active={view}
              onChange={(id) => setView(id as 'appeals' | 'info')}
              segments={[
                { id: 'appeals', label: "Консультация", icon: <MessageSquare size={14} /> },
                { id: 'info', label: t('appeal.info'), icon: <Info size={14} /> },
              ]}
            />
            <Button
              variant="primary"
              size="md"
              onClick={openCreate}
              icon={<Plus size={16} />}
              className="bg-purple-600 hover:bg-purple-700"
            >
              {t('scenario.consulting.addAnswer')}
            </Button>
          </div>
        }
      />

      <Window className="flex-1 overflow-hidden" noPadding>
        {loading ? (
          <PageLoader text={t('loader.loading')} />
        ) : view === 'info' ? (
          <InfoTab line={line} appeals={appeals} />
        ) : (
          <div className="flex flex-col h-full overflow-hidden bg-white">
            {/* Consultation Content Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              <div className="grid grid-cols-12 gap-6">
                <div className="col-span-8 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-[length:var(--text-sm)] font-semibold text-gray-900 flex items-center gap-2">
                      <MessageSquare size={16} className="text-purple-600" />
                      Текст консультации
                    </h3>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        icon={<Sparkles size={14} />}
                        className="text-purple-600 hover:bg-purple-50"
                      >
                        Спросить Laila
                      </Button>
                      <Button
                        variant="primary"
                        size="sm"
                        onClick={handleSaveText}
                        loading={isSaving}
                        icon={<Save size={14} />}
                        className="bg-purple-600"
                      >
                        Сохранить
                      </Button>
                    </div>
                  </div>
                  <Textarea
                    className="min-h-[300px] text-[length:var(--text-sm)] leading-relaxed border-purple-100 focus:border-purple-300 focus:ring-purple-200"
                    placeholder="Введите текст консультации, выводы и рекомендации..."
                    value={localText}
                    onChange={(e) => setLocalText(e.target.value)}
                  />
                </div>

                <div className="col-span-4 space-y-6">
                  <div className="p-4 rounded-xl bg-purple-50 border border-purple-100 space-y-4">
                    <div className="space-y-1">
                      <label className="text-[11px] font-bold text-purple-700 uppercase tracking-wider">О чем консультация</label>
                      <Select
                        value={localTopic}
                        onChange={setLocalTopic}
                        options={[
                          { value: 'general', label: 'Общий вопрос' },
                          { value: 'tax', label: 'Налоги' },
                          { value: 'civil', label: 'Гражданское право' },
                          { value: 'criminal', label: 'Уголовные риски' },
                        ]}
                      />
                    </div>
                  </div>

                  <div className="space-y-3">
                    <h4 className="text-[length:var(--text-xs)] font-bold text-gray-500 uppercase tracking-wider px-1">Документы и приложения</h4>
                    <div className="rounded-xl border border-gray-100 overflow-hidden bg-gray-50/50">
                      <AppealsTable
                        items={appeals}
                        onEdit={openEdit}
                        onDelete={setDeleteTarget}
                        onNavigate={onNavigateAppeal}
                        compact
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </Window>

      <AppealEditModal
        open={isNew || !!editingAppeal}
        isNew={isNew}
        isSaving={isSaving}
        formData={formData}
        setFormData={setFormData}
        onClose={closeModal}
        onSave={handleSaveAppeal}
        titleOverride={isNew ? "Новый ответ" : "Редактировать ответ"}
        subtitleOverride="Материалы и ответы на вопросы консультации"
      />

      <DeleteConfirmModal
        target={deleteTarget}
        entityLabel="ответ"
        onCancel={() => setDeleteTarget(null)}
        onConfirm={handleDeleteAppeal}
      />
    </div>
  );
}
