// types/index.ts

export type LailaMode = 'chat' | 'analyze' | 'status';
export type MessageRole = 'user' | 'laila' | 'system';
export type OrbSize = 'sm' | 'md' | 'lg' | 'fab';
export type ModuleStatus = 'online' | 'offline' | 'standby';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  text: string;
  timestamp: Date;
  isTyping?: boolean;
}

export interface AnalysisResult {
  summary: string;
  keyPoints: string[];
  risks: string[];
  articles: string[];
  confidence: number; // 0–100
}

export interface LailaSystemModule {
  id: string;
  label: string;
  status: ModuleStatus;
  description: string;
  load?: number; // 0–100
}
