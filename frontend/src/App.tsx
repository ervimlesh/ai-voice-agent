import { AppShell } from './components/app-shell';
import { VoiceAgent } from './features/voice-agent/voice-agent';

export default function App() {
  return (
    <AppShell>
      <VoiceAgent />
    </AppShell>
  );
}
