import { useCallback, useRef, useState } from 'react'

interface UseRecorderResult {
  isRecording: boolean
  error: string | null
  startRecording: () => Promise<void>
  stopRecording: () => Promise<Blob>
}

// Browser MediaRecorder defaults to webm/opus on Chrome/Edge, which is what
// backend/app/services/asr_service.py needs to handle correctly — flagged in
// the plan as something to verify for real, not assume works.
export function useRecorder(): UseRecorderResult {
  const [isRecording, setIsRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  const startRecording = useCallback(async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      chunksRef.current = []

      const recorder = new MediaRecorder(stream)
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }
      mediaRecorderRef.current = recorder
      recorder.start()
      setIsRecording(true)
    } catch {
      setError('Microphone access denied or unavailable.')
    }
  }, [])

  const stopRecording = useCallback((): Promise<Blob> => {
    return new Promise((resolve) => {
      const recorder = mediaRecorderRef.current
      if (!recorder) {
        resolve(new Blob())
        return
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType })
        streamRef.current?.getTracks().forEach((track) => track.stop())
        setIsRecording(false)
        resolve(blob)
      }
      recorder.stop()
    })
  }, [])

  return { isRecording, error, startRecording, stopRecording }
}
