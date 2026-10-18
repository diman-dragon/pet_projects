// src/pages/CaseDetails/types.ts
import { Case as GlobalCase, Line as GlobalLine, ID as GlobalID } from '../../types/entities';

export type Case = GlobalCase;
export type Line = GlobalLine;
export type ID = GlobalID;

export type CaseDetailsTab = 'lines' | 'info';
