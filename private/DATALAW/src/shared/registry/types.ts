// src/shared/registry/types.ts

import { ReactNode, ComponentType } from 'react';

/** * Минимальный контракт для любой сущности в реестре.
 * Изменили id на string | number для поддержки UUID и автоинкремента.
 */
export interface BaseEntity {
  id: string | number;
  created_at?: string | null;
  updated_at?: string | null;
}

/** Одна колонка таблицы */
export interface ColumnDef<T extends BaseEntity> {
  header: string;
  render: (item: T, ctx: ColumnContext<T>) => ReactNode;
  width?: string;
}

/** Контекст, пробрасываемый в рендер колонки */
export interface ColumnContext<T extends BaseEntity> {
  onEdit: (item: T) => void;
  onDelete: (item: T) => void;
  // Оставляем item для максимальной гибкости
  onNavigate?: (item: T) => void;
}

/** Карточка в блоке аналитики */
export interface StatCard {
  icon: ReactNode;
  value: number | string;
  label: string;
  color?: 'blue' | 'green' | 'purple' | 'amber' | 'red' | 'default';
}

/** Конфиг аналитики для конкретного ресурса */
export interface AnalyticsConfig<T extends BaseEntity> {
  stats: (items: T[]) => StatCard[];
  chartData?: (items: T[]) => { name: string; value: number }[];
  chartLabel?: string;
}

/** Полный конфиг ресурса — передаётся в RegistryPage */
export interface RegistryConfig<T extends BaseEntity> {
  title: string;
  icon: ReactNode;
  dbKey: string;
  tableName: string;

  /** Колонки для таблицы */
  columns: (ctx: ColumnContext<T>) => ColumnDef<T>[];

  /** * Форма редактирования/создания.
   */
  EditForm: ComponentType<{
    formData: any;
    setFormData: (d: any) => void;
    isNew?: boolean;
  }>;

  /** Пустые данные формы */
  defaultFormData: Partial<T> & Record<string, any>;

  /** Аналитика */
  analytics?: AnalyticsConfig<T>;

  /** Строка поиска */
  searchFn: (item: T, query: string) => boolean;

  /** SQL SELECT */
  selectSql?: string;

  /** SQL INSERT */
  insertSql: (data: any, extra?: any) => [string, any[]] | Promise<[string, any[]]>;

  /** SQL UPDATE - ID теперь тоже поддерживает строку */
  updateSql: (data: any, id: string | number) => [string, any[]] | Promise<[string, any[]]>;

  /** Имя сущности (для диалога удаления) */
  deleteType: string;

  /** Заголовок кнопки создания */
  createTitle: string;

  /** Заголовок модалки редактирования */
  editTitle: string;
}
