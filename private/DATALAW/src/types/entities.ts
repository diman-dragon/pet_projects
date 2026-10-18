import type { ComponentType, ReactNode } from 'react';

export type ID = number | string;

export interface BaseEntity {
  id: number;
  created_at?: string | null;
  updated_at?: string | null;
}

export type UserRole = 'SUPERADMIN' | 'ADMIN' | 'LAWYER' | 'ASSISTANT' | 'ANALYST';

export interface UserProfile {
  id: number;
  login: string;
  full_name: string;
  role: UserRole | string;
}

export interface StaffMember extends BaseEntity {
  full_name: string;
  login: string;
  role: UserRole | string;
  email?: string | null;
  password_hash?: string;
  salt?: string;
  avatar_url?: string | null;
  last_login?: string | null;
  is_active?: boolean;
}

export type ClientType = 'LEGAL' | 'INDIVIDUAL';
export type ClientStatus = 'ACTIVE' | 'ARCHIVE' | 'LEAD';

export interface Client extends BaseEntity {
  full_name: string;
  folder_name: string;
  email?: string | null;
  phone?: string | null;
  tax_id?: string | null;
  type?: ClientType | string;
  status?: ClientStatus | string;
  metadata?: string | null;
  client_data?: string | null;
  summary?: string | null;
}

export type ClientFull = Client;

export type CaseStatus = 'ACTIVE' | 'DONE' | 'ARCHIVE' | 'PAUSED';

export interface Case extends BaseEntity {
  client_id: number;
  title: string;
  status: CaseStatus | string;
  folder_name: string;
  case_number?: string | null;
  case_type?: string | null;
  litigation_subtype?: string | null;
  counterparty?: string | null;
  metadata?: string | null;
  summary?: string | null;
  // Scenario specific fields
  contract_date?: string | null;
  contract_type?: string | null;
  contract_amount?: string | null;
  metadata_description?: string | null;
  mandate_check?: string | null;
  mandate_notes?: string | null;
  violation_type?: string | null;
  risk_level?: string | null;
  authority_involved?: string | null;
  jurisdiction_id?: number | null;
  erasure_candidate_at?: string | null;
}

export interface Subject extends BaseEntity {
  display_code: string;
  pii_ref: string;
}

export interface CaseSubject extends BaseEntity {
  case_id: number;
  subject_id: number;
  role: string;
  risk_level?: string | null;
  consent_ref?: string | null;
}

export interface CaseDocument extends BaseEntity {
  case_id: number;
  subject_id?: number | null;
  category: string;
  file_name: string;
  file_path: string;
  file_hash: string;
  hash_algo?: string | null;
  collected_at?: string | null;
  summary?: string | null;
  category_suggested?: string | null;
  risk_level_suggested?: string | null;
}

export type CaseFull = Case;

export interface Line extends BaseEntity {
  case_id: number;
  title: string;
  position_index: number;
  folder_name: string;
  description?: string | null;
  summary?: string | null;
  status?: string | null;
  // Scenario specific
  consultation_text?: string | null;
  consultation_topic?: string | null;
  contract_analysis?: string | null;
  contract_risks?: string | null;
}

export interface Appeal extends BaseEntity {
  line_id: number;
  title: string;
  status?: string | null;
  description?: string | null;
  file_path?: string | null;
  folder_name: string;
  agency_code?: string | null;
  appeal_date?: string | null;
  recipient?: string | null;
  incoming_number?: string | null;
  outgoing_number?: string | null;
  summary?: string | null;
}

export interface AppealDocument extends BaseEntity {
  appeal_id: number;
  file_name: string;
  file_path: string;
  file_hash: string;
  summary?: string | null;
  key_points?: string | null;
  duplicate_of_path?: string | null;
  duplicate_appeal_id?: number | null;
}

export interface EntityStats {
  label: string;
  value: number;
  icon?: ComponentType<{ className?: string }> | ReactNode;
  color?: 'blue' | 'amber' | 'purple' | 'green' | 'red' | string;
}

export interface ChartDataItem {
  name: string;
  value: number;
}
