import { useCallback, useRef, useState, useEffect } from 'react';

export type HandsFreeState = 'idle' | 'requesting-permission' | 'recording' | 'processing' | 'error';

interface HandsFreeRecorderOptions {
  silenceThreshold?: number;
  minRecordingDuration?: number;
  onSilenceDetected?: () => void;
  onAudioReady?: (blob: Blob) => void;
  autoStopOnSilence?: boolean;
}

export function useHandsFreeRecorder(options: HandsFreeRecorderOptions = {}) {
  const {
    silenceThreshold = 2000,
    minRecordingDuration = 500,
    onSilenceDetected,
    onAudioReady,
    autoStopOnSilence = true,
  } = options;

  const [state, setState] = useState<HandsFreeState>('idle');
  const [error, setError] = useState<string>('');
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const silenceCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const recordingStartTimeRef = useRef<number>(0);
  const lastSoundTimeRef = useRef<number>(0);
  const silenceDetectedRef = useRef<boolean>(false);
  const onAudioReadyRef = useRef<((blob: Blob) => void) | undefined>(onAudioReady);
  const autoStopOnSilenceRef = useRef<boolean>(autoStopOnSilence);

  // Update the refs whenever callbacks/options change
  useEffect(() => {
    onAudioReadyRef.current = onAudioReady;
  }, [onAudioReady]);

  useEffect(() => {
    autoStopOnSilenceRef.current = autoStopOnSilence;
  }, [autoStopOnSilence]);

  const stopRecording = useCallback(async (): Promise<Blob | null> => {
    console.log('🛑 Stopping recording...');
    
    // Clear silence detection
    if (silenceCheckIntervalRef.current) {
      clearInterval(silenceCheckIntervalRef.current);
      silenceCheckIntervalRef.current = null;
    }

    const recorder = mediaRecorderRef.current;

    if (!recorder || recorder.state === 'inactive') {
      console.log('⚠️ Recorder already inactive');
      setState('idle');
      return null;
    }

    setState('processing');

    return new Promise((resolve) => {
      recorder.onstop = () => {
        console.log('✅ Recording stopped, creating blob');
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
        resolve(blob);
      };

      recorder.stop();
    });
  }, []);

  const startSilenceDetection = useCallback(() => {
    if (!analyserRef.current) {
      console.log('❌ Analyser not available');
      return;
    }

    console.log('🎙️ Starting silence detection...');
    silenceDetectedRef.current = false;
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    const SILENCE_THRESHOLD = 30;

    silenceCheckIntervalRef.current = setInterval(() => {
      if (!analyserRef.current || silenceDetectedRef.current) {
        return;
      }

      try {
        analyserRef.current.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length;

        if (average > SILENCE_THRESHOLD) {
          lastSoundTimeRef.current = Date.now();
        }

        const timeSinceLast = Date.now() - lastSoundTimeRef.current;
        const recordingDuration = Date.now() - recordingStartTimeRef.current;

        if (recordingDuration > minRecordingDuration && timeSinceLast > silenceThreshold) {
          console.log(`🔇 Silence detected after ${timeSinceLast}ms`);
          silenceDetectedRef.current = true;
          
          clearInterval(silenceCheckIntervalRef.current);
          silenceCheckIntervalRef.current = null;
          
          if (autoStopOnSilenceRef.current) {
            console.log('🛑 Auto-stopping recording due to silence (hands-free mode)');
            stopRecording().then((blob: Blob | null) => {
              if (blob && onAudioReadyRef.current) {
                console.log('📢 Calling onAudioReady from silence detection');
                onAudioReadyRef.current(blob);
              }
            });
          }
          
          if (onSilenceDetected) {
            console.log('📢 Calling onSilenceDetected');
            onSilenceDetected();
          }
        }
      } catch (err) {
        console.error('Error in silence detection:', err);
      }
    }, 100);
  }, [silenceThreshold, minRecordingDuration, onSilenceDetected, stopRecording]);

  const startRecording = useCallback(async () => {
    console.log('🎤 Starting recording...');
    setError('');
    setState('requesting-permission');

    try {
      // OFF for multi-voice capture: with these ON the browser cancels any
      // audio also coming from this machine's speakers (ChatGPT's voice).
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      streamRef.current = stream;
      chunksRef.current = [];
      recordingStartTimeRef.current = Date.now();
      lastSoundTimeRef.current = Date.now();
      silenceDetectedRef.current = false;

      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;
      const analyser = audioContext.createAnalyser();
      analyserRef.current = analyser;
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : '';

      const recorder = new MediaRecorder(stream, {
        ...(mimeType ? { mimeType } : {}),
        audioBitsPerSecond: 192000,
      });

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.start(100);
      mediaRecorderRef.current = recorder;
      setState('recording');
      console.log('🔴 Recording active');

      startSilenceDetection();
    } catch (err) {
      console.error('Error starting recording:', err);
      setError(err instanceof Error ? err.message : 'Unable to access microphone.');
      setState('error');
    }
  }, [startSilenceDetection]);

  const resetError = useCallback(() => {
    setError('');
    setState('idle');
  }, []);

  const cleanup = useCallback(() => {
    console.log('🧹 Cleaning up...');
    if (silenceCheckIntervalRef.current) {
      clearInterval(silenceCheckIntervalRef.current);
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
