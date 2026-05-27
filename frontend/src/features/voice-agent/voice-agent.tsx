import { FormEvent, useMemo, useState, useRef, useEffect, useCallback } from 'react';
import { Loader2, Mic, Send, Square, Trash2, Volume2, Pause, Play, Radio, Plus, X } from 'lucide-react';

import { askByText, askByVoice } from '../../api/agent-api';
import { useHandsFreeRecorder } from '../../hooks/use-hands-free-recorder-fixed';
import { useWebSocketRecorder } from '../../hooks/use-websocket-recorder';
import type { ChatMessage } from '../../types/chat';

type SpeakerRoleLabel = 'Doctor' | 'Patient' | 'Relative' | 'Unknown';

interface MessageWithSpeaker extends ChatMessage {
  speakerRole?: SpeakerRoleLabel;
  speakerId?: string;          // stable identity e.g. "S1"
  speakerConfidence?: number;  // 0-100 (role confidence)
  overlap?: boolean;           // turn flagged as overlapping speech
  ragSuggestions?: string[];
  pending?: boolean;
  turnId?: string;             // backend-coalesced bubble identity (continuous-speaker)
  closed?: boolean;            // backend has sealed this bubble (turn_close received)
}

// Map a speaker role to a CSS modifier class for bubble coloring.
function speakerClass(role?: SpeakerRoleLabel): string {
  switch (role) {
    case 'Doctor':
      return 'speaker-doctor';
    case 'Patient':
      return 'speaker-patient';
    case 'Relative':
      return 'speaker-relative';
    default:
      return 'speaker-unknown';
  }
}

// Normalize a question for matching: lowercase, strip punctuation, collapse whitespace.
function normalizeQuestion(q: string): string {
  return q
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// Loose overlap match: spoken text "contains" the suggestion or shares enough
// significant tokens. Returns true if the suggestion should be considered "asked".
function isSuggestionAsked(suggestion: string, spoken: string): boolean {
  const s = normalizeQuestion(suggestion);
  const t = normalizeQuestion(spoken);
  if (!s || !t) return false;
  if (t.includes(s) || s.includes(t)) return true;

  const stop = new Set([
    'a','an','the','is','are','was','were','do','does','did','you','your','i','me','my',
    'we','us','our','to','of','in','on','for','at','and','or','but','have','has','had',
    'be','been','being','it','this','that','these','those','as','with','about','any',
    'how','what','when','where','why','which','who','can','could','would','should','will',
  ]);
  const sigTokens = (str: string) =>
    str.split(' ').filter((w) => w.length > 2 && !stop.has(w));
  const sTok = sigTokens(s);
  const tTok = new Set(sigTokens(t));
  if (sTok.length === 0) return false;
  const overlap = sTok.filter((w) => tTok.has(w)).length;
  return overlap / sTok.length >= 0.7;
}

export function VoiceAgent() {
  const [history, setHistory] = useState<MessageWithSpeaker[]>([]);
  const [textMessage, setTextMessage] = useState('');
  const [status, setStatus] = useState('🎤 Ready. Speak in any language.');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isHandsFreeMode, setIsHandsFreeMode] = useState(false);
  // Sidebar question list — accumulates suggestions generated for both Doctor
  // and Patient turns. Items are removed once asked (via voice or click) or
  // dismissed via the cross icon.
  const [sidebarSuggestions, setSidebarSuggestions] = useState<string[]>([]);
  const lastAudioBlobRef = useRef<Blob | null>(null);
  const conversationPanelRef = useRef<HTMLDivElement>(null);

  // Auto-scroll conversation panel
  useEffect(() => {
    if (conversationPanelRef.current) {
      conversationPanelRef.current.scrollTop = conversationPanelRef.current.scrollHeight;
    }
  }, [history]);

  // Add suggestions to sidebar, deduped against what's already there and against
  // any questions that have already been spoken in the conversation.
  const addSuggestionsToSidebar = useCallback(
    (incoming: string[] | undefined, spokenSoFar: string[]) => {
      if (!incoming || incoming.length === 0) return;
      setSidebarSuggestions((prev) => {
        const existingNorm = new Set(prev.map(normalizeQuestion));
        const next = [...prev];
        for (const q of incoming) {
          const n = normalizeQuestion(q);
          if (!n || existingNorm.has(n)) continue;
          // Skip if it was already asked at any point in the conversation.
          if (spokenSoFar.some((spoken) => isSuggestionAsked(q, spoken))) continue;
          existingNorm.add(n);
          next.push(q);
        }
        return next;
      });
    },
    []
  );

  // Remove any sidebar suggestion that matches the spoken text.
  const removeAskedSuggestions = useCallback((spoken: string) => {
    setSidebarSuggestions((prev) => prev.filter((q) => !isSuggestionAsked(q, spoken)));
  }, []);

  // ─── WebSocket-based hands-free recorder (Silero VAD) ───
  // Continuous-speaker bubble flow:
  //   turn_new    → append a new bubble keyed by turn_id
  //   turn_update → find the bubble by turn_id and replace its text/end
  //   turn_close  → mark the bubble as sealed (no future updates)
  // The legacy `speaker_turns` event still fires (back-compat) but we now use
  // it only to drive sidebar suggestion removal — bubble lifecycle is owned by
  // the three turn_* events.
  const wsRecorder = useWebSocketRecorder({
    onTurnNew: (turn) => {
      setHistory((prev) => [
        ...prev,
        {
          role: 'user',
          content: turn.text,
          speakerRole: (turn.role as SpeakerRoleLabel) || 'Unknown',
          speakerId: turn.speaker_id,
          speakerConfidence: turn.role_confidence,
          overlap: turn.overlap,
          turnId: turn.turn_id,
        },
      ]);
      removeAskedSuggestions(turn.text);
    },
    onTurnUpdate: (turn) => {
      setHistory((prev) => {
        const idx = prev.findIndex((m) => m.turnId === turn.turn_id);
        if (idx < 0) return prev;
        const next = prev.slice();
        next[idx] = {
          ...next[idx],
          content: turn.text,
          speakerRole: (turn.role as SpeakerRoleLabel) || next[idx].speakerRole,
          speakerConfidence: turn.role_confidence,
          overlap: turn.overlap,
        };
        return next;
      });
      removeAskedSuggestions(turn.text);
    },
    onTurnClose: (turn) => {
      setHistory((prev) => {
        const idx = prev.findIndex((m) => m.turnId === turn.turn_id);
        if (idx < 0) return prev;
        const next = prev.slice();
        next[idx] = { ...next[idx], closed: true };
        return next;
      });
    },
    onSpeakerTurns: (turns, _language) => {
      // Bubble lifecycle is handled by turn_new/turn_update — here we only
      // update status text and prune sidebar suggestions.
      if (!turns || turns.length === 0) return;
      for (const t of turns) removeAskedSuggestions(t.text);
      const summary = turns.map((t) => `${t.speaker_id}·${t.role}`).join(', ');
      setStatus(`🗣️ ${turns.length} turn(s): ${summary}`);
    },
    onTranscript: (text, _language, isValid) => {
      // Kept as a fallback for the '[No valid speech]' invalid case; valid
      // transcripts now arrive via onSpeakerTurns, so ignore valid ones here to
      // avoid duplicate bubbles.
      if (isValid) return;
      removeAskedSuggestions(text);
    },
    onResponse: (transcript, _reply, speakerRole, _speakerConfidence, ragSuggestions) => {
      // Turns are already rendered via onSpeakerTurns. Here we only feed the
      // sidebar with doctor-facing suggestions for the driving (last patient)
      // turn, and discard the AI reply text (we don't show AI answers).
      setHistory((prev) => {
        const spokenSoFar = prev
          .filter((m) => m.role === 'user')
          .map((m) => m.content);
        addSuggestionsToSidebar(ragSuggestions, spokenSoFar);
        return prev;
      });
      if (transcript) removeAskedSuggestions(transcript);

      const speakerDisplay = speakerRole ? `${speakerRole} spoke` : 'Captured';
      setStatus(`🎤 ${speakerDisplay}. Listening for next speech...`);
    },
    onVADEvent: (event, confidence) => {
      if (event === 'speech_start') {
        setStatus(`🔴 Speech detected (${(confidence * 100).toFixed(0)}%) - listening...`);
      } else if (event === 'speech_end') {
        setStatus('⏳ Processing your speech...');
      }
    },
    onStatusChange: (newStatus) => {
      setStatus(newStatus);
    },
    onError: (err) => {
      console.error('WS Recorder error:', err);
    },
  });

  // ─── Legacy manual recorder (for single-shot record & process) ───
  const handleAudioReady = useCallback((audioBlob: Blob) => {
    lastAudioBlobRef.current = audioBlob;
  }, []);

  const {
    state: manualState,
    error: manualError,
    isRecording: isManualRecording,
    startRecording,
    stopRecording,
    resetError,
    cleanup: manualCleanup,
  } = useHandsFreeRecorder({
    silenceThreshold: 3500,
    minRecordingDuration: 1000,
    autoStopOnSilence: false,
    onAudioReady: handleAudioReady,
  });

  // ─── Derived status mode ───
  const statusMode = useMemo(() => {
    if (isHandsFreeMode) {
      if (wsRecorder.isProcessing) return 'Processing';
      if (wsRecorder.vadStatus.speaking) return 'Listening';
      if (wsRecorder.isRecording) return 'Recording';
      if (wsRecorder.state === 'connecting') return 'Connecting';
      return 'Hands-Free';
    }
    if (isProcessing) return 'Processing';
    if (isManualRecording) return 'Recording';
    if (manualState === 'requesting-permission') return 'Permission';
    if (manualError || wsRecorder.error) return 'Error';
    return 'Idle';
  }, [isHandsFreeMode, wsRecorder, isProcessing, isManualRecording, manualState, manualError]);

  // ─── Manual record handlers ───
  async function handleStartRecording() {
    resetError();
    lastAudioBlobRef.current = null;
    setStatus('🔴 Recording... Speak now.');
    await startRecording();
  }

  async function handleStopRecording() {
    setIsProcessing(true);
    setStatus('⏹️ Processing audio...');

    try {
      let audioBlob = lastAudioBlobRef.current;
      if (!audioBlob) {
        audioBlob = await stopRecording();
      } else {
        await stopRecording();
      }

      if (!audioBlob || audioBlob.size === 0) {
        setStatus('❌ No audio captured. Try again.');
        return;
      }

      setStatus('🎯 Transcribing & generating response...');
      const response = await askByVoice(audioBlob, history);
      // Update history but strip any assistant replies — we only show context.
      setHistory(response.history as MessageWithSpeaker[]);
      // Pull suggestions out of the new history and merge into the sidebar.
      const spoken = (response.history as MessageWithSpeaker[])
        .filter((m) => m.role === 'user')
        .map((m) => m.content);
      const incoming = (response.history as MessageWithSpeaker[])
        .flatMap((m) => m.ragSuggestions ?? []);
      addSuggestionsToSidebar(incoming, spoken);
      setStatus('✅ Context captured.');
    } catch (err) {
      setStatus(err instanceof Error ? `❌ ${err.message}` : '❌ Something went wrong.');
    } finally {
      setIsProcessing(false);
    }
  }

  // ─── Hands-free handlers (WebSocket + Silero VAD) ───
  async function handleStartHandsFree() {
    resetError();
    setIsHandsFreeMode(true);
    setStatus('🔌 Connecting to voice stream...');
    await wsRecorder.startHandsFree(history);
  }

  function handleStopHandsFree() {
    wsRecorder.stopHandsFree();
    setIsHandsFreeMode(false);
    setStatus('⏹️ Hands-free mode stopped.');
  }

  // ─── Text submit ───
  async function handleTextSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = textMessage.trim();
    if (!message) {
      setStatus('⚠️ Please type a message.');
      return;
    }

    setIsProcessing(true);
    setStatus('🤔 Capturing context...');

    try {
      const response = await askByText(message, history);
      setHistory(response.history as MessageWithSpeaker[]);
      const spoken = (response.history as MessageWithSpeaker[])
        .filter((m) => m.role === 'user')
        .map((m) => m.content);
      const incoming = (response.history as MessageWithSpeaker[])
        .flatMap((m) => m.ragSuggestions ?? []);
      addSuggestionsToSidebar(incoming, spoken);
      removeAskedSuggestions(message);
      setTextMessage('');
      setStatus('✅ Context captured.');
    } catch (err) {
      setStatus(err instanceof Error ? `❌ ${err.message}` : '❌ Something went wrong.');
    } finally {
      setIsProcessing(false);
    }
  }

  // ─── Suggestion handlers ───
  // Clicking a sidebar question means the Doctor is about to ask it. We insert
  // it into the chat as a Doctor bubble (acting as context) and remove it from
  // the sidebar. The voice transcript that follows will simply match and be a
  // no-op for the sidebar.
  const handleSuggestionClick = useCallback((question: string) => {
    setHistory((prev) => [
      ...prev,
      { role: 'user', content: question, speakerRole: 'Doctor' },
    ]);
    setSidebarSuggestions((prev) => prev.filter((q) => q !== question));
  }, []);

  const handleSuggestionDismiss = useCallback(
    (e: React.MouseEvent, question: string) => {
      e.stopPropagation();
      setSidebarSuggestions((prev) => prev.filter((q) => q !== question));
    },
    []
  );

  // ─── Clear chat ───
  function clearChat() {
    setHistory([]);
    setTextMessage('');
    setSidebarSuggestions([]);
    if (isHandsFreeMode) {
      wsRecorder.resetHistory();
    }
    setStatus('🧹 Conversation reset.');
  }

  // ─── Cleanup on unmount ───
  useEffect(() => {
    return () => {
      manualCleanup();
      wsRecorder.cleanup();
    };
  }, [manualCleanup, wsRecorder.cleanup]);

  const anyProcessing = isProcessing || wsRecorder.isProcessing;
  const anyRecording = isManualRecording || wsRecorder.isRecording;

  return (
    <section className="agent-card">
      {/* Header */}
      <div className="agent-header">
        <div className={`status-indicator status-${statusMode.toLowerCase()}`}>
          {isHandsFreeMode && wsRecorder.vadStatus.speaking ? (
            <div className="status-icon recording">
              <div className="pulse"></div>
              <div className="pulse"></div>
            </div>
          ) : statusMode === 'Processing' ? (
            <div className="status-icon processing">
              <div className="spinner"></div>
            </div>
          ) : (
            <div className="status-icon idle">
              <div className="indicator-dot"></div>
            </div>
          )}
        </div>
      </div>

      {/* Status Bar */}
      <div className="status-box">
        <Volume2 size={18} />
        <span>{manualError || wsRecorder.error || status}</span>
      </div>

      {/* VAD Indicator */}
      {isHandsFreeMode && (
        <div className="vad-indicator">
          <div
            className="vad-bar"
            style={{
              width: `${Math.max(5, wsRecorder.vadStatus.confidence * 100)}%`,
              background: wsRecorder.vadStatus.speaking
                ? 'linear-gradient(90deg, #ef4444, #f87171)'
                : 'linear-gradient(90deg, #14b8a6, #06b6d4)',
              transition: 'width 0.15s ease, background 0.3s ease',
            }}
          />
        </div>
      )}

      {/* Main Content Wrapper */}
      <div className="main-content-wrapper">
        {/* Chat Area (Left) — context only, no AI replies */}
        <div className="conversation-panel" ref={conversationPanelRef}>
          {history.length === 0 ? (
            <div className="empty-state">
              <div>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🎙️</div>
                <div style={{ fontSize: '0.9rem', marginTop: '0.5rem', opacity: 0.7 }}>
                  Communicate in Hindi, Telugu, English, or any known human language!
                </div>
              </div>
            </div>
          ) : (
            history.map((message, index) => {
              // Skip assistant messages entirely — we only render spoken/typed
              // questions as context.
              if (message.role !== 'user') return null;

              const role = message.speakerRole || 'Unknown';
              // role_confidence arrives 0-100 from the backend turns.
              const confPct =
                typeof message.speakerConfidence === 'number'
                  ? Math.round(message.speakerConfidence)
                  : undefined;
              const label = `${message.speakerId ? `${message.speakerId} · ` : ''}${role}${
                confPct !== undefined ? ` (${confPct}%)` : ''
              }`;

              return (
                <article
                  key={message.turnId ?? `${message.role}-${index}`}
                  className={`message message-${message.role} ${speakerClass(role)}${
                    message.closed ? ' message-closed' : ''
                  }`}
                >
                  <div className="message-role">
                    {label}
                    {message.overlap && (
                      <span className="overlap-badge" title="Overlapping speech detected">
                        {' '}⚠ overlap
                      </span>
                    )}
                  </div>
                  <p>{message.content}</p>
                </article>
              );
            })
          )}
        </div>

        {/* Suggestions Sidebar (Right) — questions for the Doctor */}
        {sidebarSuggestions.length > 0 && (
          <div className="suggestions-sidebar-right">
            <div className="suggestions-header">
              <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#c7d2fe' }}>
                💡 Questions for Doctor
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--muted)', marginTop: '6px', lineHeight: 1.4 }}>
                Based on the conversation context
              </div>
            </div>
            <div className="suggestions-items">
              {sidebarSuggestions.map((suggestion, idx) => (
                <div
                  key={`${idx}-${suggestion}`}
                  className="suggestion-button"
                  title="Click to ask this question"
                  role="button"
                  tabIndex={0}
                  onClick={() => handleSuggestionClick(suggestion)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleSuggestionClick(suggestion);
                    }
                  }}
                >
                  <div className="suggestion-number">{idx + 1}</div>
                  <div className="suggestion-content">{suggestion}</div>
                  <button
                    type="button"
                    className="suggestion-dismiss"
                    aria-label="Dismiss question"
                    title="Dismiss"
                    onClick={(e) => handleSuggestionDismiss(e, suggestion)}
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Bottom Action Bar (Like Perplexity) */}
      <div className="perplexity-bottom-bar">
        <div className="bar-content">
          {/* Left: Hands-Free Button */}
          <div className="bar-left">
            {!isHandsFreeMode ? (
              <button
                className="icon-button hands-free-btn"
                onClick={handleStartHandsFree}
                disabled={isManualRecording || isProcessing}
                title="Start hands-free mode (Microphone)"
              >
                <Radio size={20} />
              </button>
            ) : (
              <button
                className="icon-button hands-free-btn active"
                onClick={handleStopHandsFree}
                title="Stop hands-free mode"
              >
                <Pause size={20} />
              </button>
            )}
          </div>

          {/* Center: Conversation Guide */}
          <div className="bar-center">
            <div className="bar-hint">
              {isHandsFreeMode
                ? '🎤 Hands-Free Mode Active - Speak now'
                : '💬 Ready for conversation'}
            </div>
          </div>

          {/* Right: Additional Controls */}
          <div className="bar-right">
            <button
              className="icon-button new-chat-btn"
              onClick={clearChat}
              disabled={anyProcessing}
              title="Reset conversation"
            >
              <Plus size={20} />
            </button>
            <button
              className="icon-button reset-btn"
              onClick={clearChat}
              disabled={anyProcessing}
              title="Clear chat"
            >
              <Trash2 size={20} />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
