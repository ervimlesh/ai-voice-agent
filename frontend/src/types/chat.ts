export type ChatRole = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  role: ChatRole;
  content: string;
  ragSuggestions?: string[];
  speakerRole?: 'Doctor' | 'Patient' | 'Relative' | 'Unknown';
  speakerConfidence?: number;
}

export interface AgentResponse {
  transcript: string;
  reply: string;
  history: ChatMessage[];
  ragSuggestions?: string[];
  speakerRole?: 'Doctor' | 'Patient' | 'Relative' | 'Unknown';
  speakerConfidence?: number;
}
