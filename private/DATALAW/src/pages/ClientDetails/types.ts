// src/pages/ClientDetails/types.ts
import {
  Case as GlobalCase,
  ClientFull as GlobalClientFull,
  CaseStatus as GlobalCaseStatus,
  ID as GlobalID,
} from '../../types/entities';

/**
 * Переиспользуем глобальные определения сущностей.
 * Это гарантирует, что детали клиента и список дел
 * всегда синхронизированы с основной базой данных.
 */

export type Case = GlobalCase;
export type CaseStatus = GlobalCaseStatus;
export type ID = GlobalID;

/**
 * ClientFull теперь полностью соответствует глобальной модели Client.
 * Данные, такие как ФИО, хранятся в full_name.
 */
export type ClientFull = GlobalClientFull;

/**
 * Вспомогательный интерфейс для состояния вкладок (если используется в логике)
 */
export type TabType = 'info' | 'cases' | 'documents';
