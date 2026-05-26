import { useCallback, useRef, useState, useEffect } from 'react';
import type { ChatMessage } from '../types/chat';

export type WSRecorderState =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'recording'
  | 'processing'
  | 'error';

export interface VADStatus {
  speaking: boolean;
  confidence: number;
}

export interface SpeakerTurn {
  speaker_id: string;          // stable across the conversation, e.g. "S1"
  role: string;                // Doctor | Patient | Relative | Unknown
  role_confidence: number;     // 0-100
  text: string;                // may be prefixed with "[overlapping speech]"
  start: number;               // seconds within the segment
  end: number;
  is_new_speaker: boolean;
  overlap?: boolean;
}

export interface WSRecorderCallbacks {
  onTranscript?: (text: string, language: string, isValid: boolean) => void;
  onResponse?: (transcript: string, reply: string, speakerRole?: string, speakerConfidence?: number, ragSuggestions?: string[]) => void;
  onSpeakerTurns?: (turns: SpeakerTurn[], language: string) => void;
  onVADEvent?: (event: string, confidence: number) => void;
  onStatusChange?: (status: string) => void;
  onError?: (error: string) => void;
}

const WS_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1')
  .replace(/^http/, 'ws');

export function useWebSocketRecorder(callbacks: WSRecorderCallbacks = {}) {
  const [state, setState] = useState<WSRecorderState>('idle');
  const [error, setError] = useState('');
  const [vadStatus, setVadStatus] = useState<VADStatus>({ speaking: false, confidence: 0 });

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const stateRef = useRef<WSRecorderState>('idle');
  const callbacksRef = useRef(callbacks);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep callbacks ref fresh
  useEffect(() => {
    callbacksRef.current = callbacks;
  }, [callbacks]);

  // Keep stateRef in sync
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  /** ---- WebSocket connection ---- */
  const connectWebSocket = useCallback((): Promise<WebSocket> => {
    return new Promise((resolve, reject) => {
      const wsUrl = `${WS_BASE_URL}/ws/audio-stream`;
      console.log('🔌 Connecting WebSocket:', wsUrl);

      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';

      const timeout = setTimeout(() => {
        ws.close();
        reject(new Error('WebSocket connection timeout'));
      }, 10000);

      ws.onopen = () => {
        clearTimeout(timeout);
        console.log('✅ WebSocket connected');
        wsRef.current = ws;
        resolve(ws);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWSMessage(data);
        } catch (e) {
          console.error('Failed to parse WS message:', e);
        }
      };

      ws.onerror = (e) => {
        clearTimeout(timeout);
        console.error('❌ WebSocket error:', e);
        reject(new Error('WebSocket connection failed'));
      };

      ws.onclose = (e) => {
        clearTimeout(timeout);
        console.log('🔌 WebSocket closed:', e.code, e.reason);
        wsRef.current = null;

        // If we were recording, transition to error so user knows
        if (stateRef.current === 'recording' || stateRef.current === 'processing') {
          setError('Connection lost. Please restart hands-free mode.');
          setState('error');
          stopAudioCapture();
        }
      };
    });
  }, []);

  /** ---- Handle incoming WS messages ---- */
  const handleWSMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'connected':
        console.log('🟢 Server confirmed connection');
        break;

      case 'session_started':
        console.log('🚀 Session started');
        break;

      case 'vad_event':
        if (data.event === 'speech_start') {
          setVadStatus({ speaking: true, confidence: data.confidence });
          callbacksRef.current.onVADEvent?.('speech_start', data.confidence);
          callbacksRef.current.onStatusChange?.('🎤 Speech detected - listening...');
        } else if (data.event === 'speech_end') {
          setVadStatus({ speaking: false, confidence: data.confidence });
          callbacksRef.current.onVADEvent?.('speech_end', data.confidence);
          callbacksRef.current.onStatusChange?.('⏳ Processing speech...');
          setState('processing');
        }
        break;

      case 'processing':
        if (data.stage === 'transcribing') {
          callbacksRef.current.onStatusChange?.('📝 Transcribing speech...');
        } else if (data.stage === 'thinking') {
          callbacksRef.current.onStatusChange?.('🤔 AI is thinking...');
        } else if (data.stage === 'skipped') {
          callbacksRef.current.onStatusChange?.('🎤 Listening... (speak now)');
          setState('recording');
        }
        break;

      case 'transcript':
        callbacksRef.current.onTranscript?.(data.text, data.language, data.is_valid);
        if (!data.is_valid) {
          callbacksRef.current.onStatusChange?.('🎤 Listening... (speak clearly)');
          setState('recording');
        }
        break;

      case 'speaker_turns':
        console.log(`🗣️ ${data.turns?.length ?? 0} speaker turn(s):`, data.turns);
        callbacksRef.current.onSpeakerTurns?.(data.turns ?? [], data.language);
        break;

      case 'response':
        console.log('📤 Response received:', data.transcript);
        console.log(`🎙️ Speaker: ${data.speaker_role} (confidence: ${data.speaker_confidence}%)`);
        if (data.rag_suggestions?.length > 0) {
          console.log(`🔍 RAG Suggestions:`, data.rag_suggestions);
        }
        callbacksRef.current.onResponse?.(data.transcript, data.reply, data.speaker_role, data.speaker_confidence, data.rag_suggestions);
        callbacksRef.current.onStatusChange?.('🎤 Listening for next speech...');
        setState('recording');
        break;

      case 'ready_for_next':
        console.log('✅ Backend ready for next speech');
        callbacksRef.current.onStatusChange?.('🎤 Listening for next speech...');
        setState('recording');
        break;

      case 'error':
        console.error('Server error:', data.message);
        callbacksRef.current.onError?.(data.message);
        callbacksRef.current.onStatusChange?.(`❌ ${data.message}`);
        // Resume recording after error so we can retry
        setState('recording');
        break;

      case 'keepalive':
        break;

      case 'history_synced':
        console.log('📚 History synced:', data.history_length, 'messages');
        break;

      case 'history_reset':
        console.log('🗑️ History reset');
        break;

      default:
        console.log('Unknown WS message:', data);
    }
  }, []);

  /** ---- Audio capture using ScriptProcessorNode ---- */
  const startAudioCapture = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      sourceRef.current = source;

      // Use ScriptProcessorNode to get raw PCM data
      // Buffer size 4096 at 16kHz = ~256ms chunks
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        const inputData = e.inputBuffer.getChannelData(0);
        // Convert Float32 to Int16 PCM
        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        ws.send(pcm16.buffer);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      console.log('🎙️ Audio capture started (16kHz PCM)');
    } catch (err) {
      console.error('Error starting audio capture:', err);
      throw err;
    }
  }, []);

  const stopAudioCapture = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    console.log('🔇 Audio capture stopped');
  }, []);

  /** ---- Public API ---- */

  const startHandsFree = useCallback(
    async (existingHistory?: ChatMessage[]) => {
      setError('');
      setState('connecting');

      try {
        // 1. Connect WebSocket
        const ws = await connectWebSocket();

        // 2. Start session
        ws.send(JSON.stringify({ type: 'start_session' }));

        // 3. Sync history if provided
        if (existingHistory && existingHistory.length > 0) {
          ws.send(
            JSON.stringify({
              type: 'set_history',
              history: existingHistory.map((m) => ({ role: m.role, content: m.content })),
            })
          );
        }

        // 4. Start audio capture
        await startAudioCapture();

        setState('recording');
        callbacksRef.current.onStatusChange?.('🎤 Hands-free mode active. Speak now...');
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Failed to start hands-free mode';
        setError(msg);
        setState('error');
        callbacksRef.current.onError?.(msg);
        cleanup();
      }
    },
    [connectWebSocket, startAudioCapture]
  );

  const stopHandsFree = useCallback(() => {
    // Send end_session command
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'end_session' }));
      ws.close(1000, 'User stopped hands-free mode');
    }
    wsRef.current = null;

    stopAudioCapture();
    setState('idle');
    setVadStatus({ speaking: false, confidence: 0 });
    callbacksRef.current.onStatusChange?.('⏹️ Hands-free mode stopped.');
  }, [stopAudioCapture]);

  const resetHistory = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'reset_history' }));
    }
  }, []);

  const cleanup = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    stopAudioCapture();
    const ws = wsRef.current;
    if (ws) {
      ws.close(1000, 'Cleanup');
      wsRef.current = null;
    }
    setState('idle');
    setVadStatus({ speaking: false, confidence: 0 });
  }, [stopAudioCapture]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  return {
    state,
    error,
    vadStatus,
    isRecording: state === 'recording',
    isProcessing: state === 'processing',
    isConnected: state === 'recording' || state === 'processing' || state === 'connected',
    startHandsFree,
    stopHandsFree,
    resetHistory,
    cleanup,
    resetError: useCallback(() => {
      setError('');
      setState('idle');
    }, []),
  };
}
