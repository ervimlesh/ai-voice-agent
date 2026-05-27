import { useCallback, useRef, useState } from 'react';

export type RecorderState = 'idle' | 'requesting-permission' | 'recording' | 'stopping' | 'error';

export function useAudioRecorder() {
  const [state, setState] = useState<RecorderState>('idle');
  const [error, setError] = useState<string>('');
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const startRecording = useCallback(async () => {
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

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : '';

      const recorder = new MediaRecorder(stream, {
        ...(mimeType ? { mimeType } : {}),
        audioBitsPerSecond: 192000,  // Higher bitrate for better quality
      });

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.start(100);  // More frequent chunks for better capture
      mediaRecorderRef.current = recorder;
      setState('recording');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to access microphone.');
      setState('error');
    }
  }, []);

  const stopRecording = useCallback(async (): Promise<Blob | null> => {
    const recorder = mediaRecorderRef.current;

    if (!recorder || recorder.state === 'inactive') {
      setState('idle');
      return null;
    }

    setState('stopping');

    return new Promise((resolve) => {
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        mediaRecorderRef.current = null;
        chunksRef.current = [];
        setState('idle');
        resolve(blob);
      };

      recorder.stop();
    });
  }, []);

  const resetError = useCallback(() => {
    setError('');
    setState('idle');
  }, []);

  return {
    state,
    error,
    isRecording: state === 'recording',
    startRecording,
    stopRecording,
    resetError,
  };
}
