import { apiFetch } from './http';
import type { AgentResponse, ChatMessage } from '../types/chat';

export async function askByText(message: string, history: ChatMessage[]): Promise<AgentResponse> {
  return apiFetch<AgentResponse>('/agent/text', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message, history }),
  });
}

export async function askByVoice(audioBlob: Blob, history: ChatMessage[]): Promise<AgentResponse> {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');
  formData.append('history', JSON.stringify(history));

  return apiFetch<AgentResponse>('/agent/voice', {
    method: 'POST',
    body: formData,
  });
}
