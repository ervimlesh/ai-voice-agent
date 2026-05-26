import { useCallback, useRef, useState } from 'react';

export type HandsFreeState = 'idle' | 'requesting-permission' | 'recording' | 'processing' | 'error';

interface HandsFreeRecorderOptions {
  silenceThreshold?: number;  // milliseconds of silence to trigger stop
  minRecordingDuration?: number;  // minimum recording duration before checking silence
  onSilenceDetected?: () => void;
  onAudioReady?: (blob: Blob) => void;
}

export function useHandsFreeRecorder(options: HandsFreeRecorderOptions = {}) {
  const {
    silenceThreshold = 2000,
    minRecordingDuration = 500,
    onSilenceDetected,
    onAudioReady,
  } = options;

  const [state, setState] = useState<HandsFreeState>('idle');
  const [error, setError] = useState<string>('');
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const silenceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const recordingStartTimeRef = useRef<number>(0);
  const lastSoundTimeRef = useRef<number>(0);
  const silenceCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const stopRecording = useCallback(async (): Promise<Blob | null> => {
    // Clear silence detection
    if (silenceCheckIntervalRef.current) {
      clearInterval(silenceCheckIntervalRef.current);
    }
    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current);
    }

    const recorder = mediaRecorderRef.current;

    if (!recorder || recorder.state === 'inactive') {
      setState('idle');
      return null;
    }

    setState('processing');

    return new Promise((resolve) => {
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        mediaRecorderRef.current = null;
        chunksRef.current = [];

        if (audioContextRef.current) {
          audioContextRef.current.close();
          audioContextRef.current = null;
        }

        setState('idle');

        if (onAudioReady) {
          onAudioReady(blob);
        }

        resolve(blob);
      };

      recorder.stop();
    });
  }, [onAudioReady]);

  const resetError = useCallback(() => {
    setError('');
    setState('idle');
  }, []);

  const cleanup = useCallback(() => {
    if (silenceCheckIntervalRef.current) {
      clearInterval(silenceCheckIntervalRef.current);
    }
    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current);
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
  }, []);

  return {
    state,
    error,
    isRecording: state === 'recording',
    isProcessing: state === 'processing',
    startRecording,
    stopRecording,
    resetError,
    cleanup,
  };
}
