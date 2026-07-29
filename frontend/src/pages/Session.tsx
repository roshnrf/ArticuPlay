import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { completeSession, getChild, getWords, scoreItem, speakWord, startSession, transcribe } from '../api/client'
import type { ChildRead, ScoreResult, SessionRead, WordContentRead } from '../types/api'
import { useRecorder } from '../components/useRecorder'
import { DrillWordCard } from '../components/DrillWordCard'
import { FeedbackDisplay } from '../components/FeedbackDisplay'

const ITEMS_PER_SESSION = 8
const MAX_ATTEMPTS = 3

type Phase = 'loading' | 'ready' | 'recording' | 'processing' | 'feedback' | 'error'

function shuffle<T>(arr: T[]): T[] {
  const copy = [...arr]
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[copy[i], copy[j]] = [copy[j], copy[i]]
  }
  return copy
}

export function Session() {
  const { childId } = useParams<{ childId: string }>()
  const navigate = useNavigate()
  const recorder = useRecorder()

  const [child, setChild] = useState<ChildRead | null>(null)
  const [session, setSession] = useState<SessionRead | null>(null)
  const [words, setWords] = useState<WordContentRead[]>([])
  const [itemIndex, setItemIndex] = useState(0)
  const [attemptNum, setAttemptNum] = useState(1)
  const [phase, setPhase] = useState<Phase>('loading')
  const [lastResult, setLastResult] = useState<ScoreResult | null>(null)
  const [starsEarned, setStarsEarned] = useState(0)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const initRan = useRef(false)

  useEffect(() => {
    if (!childId || initRan.current) return
    initRan.current = true

    async function init() {
      try {
        const c = await getChild(childId!)
        setChild(c)
        const s = await startSession({ child_id: childId!, language: c.language, level: c.current_level })
        setSession(s)
        const w = await getWords(c.current_level, c.language)
        setWords(shuffle(w).slice(0, ITEMS_PER_SESSION))
        setPhase('ready')
      } catch {
        setErrorMessage('Could not start the session. Please go back and try again.')
        setPhase('error')
      }
    }
    init()
  }, [childId])

  const currentWord = words[itemIndex]

  async function handlePlay() {
    if (!child || !currentWord) return
    const blob = await speakWord(currentWord.word, child.language)
    const audio = new Audio(URL.createObjectURL(blob))
    await audio.play()
  }

  async function handleRecordToggle() {
    if (!recorder.isRecording) {
      await recorder.startRecording()
      setPhase('recording')
      return
    }

    setPhase('processing')
    try {
      const blob = await recorder.stopRecording()
      const asrResult = await transcribe(blob, child!.language, currentWord.word)
      const result = await scoreItem({
        session_id: session!.id,
        item_index: itemIndex,
        target_word: currentWord.word,
        language: child!.language,
        child_transcript: asrResult.transcript,
        attempt_num: attemptNum,
        phone_classifier_flag: asrResult.phone_classifier_flag,
      })
      setLastResult(result)
      if (result.passed) setStarsEarned((s) => s + 1)
      setPhase('feedback')
    } catch {
      setErrorMessage('Something went wrong scoring that attempt. Try again.')
      setPhase('ready')
    }
  }

  async function handleContinue() {
    const shouldAdvance = lastResult?.passed || attemptNum >= MAX_ATTEMPTS

    if (!shouldAdvance) {
      setAttemptNum((a) => a + 1)
      setLastResult(null)
      setPhase('ready')
      return
    }

    const isLastItem = itemIndex + 1 >= words.length
    if (isLastItem) {
      await completeSession(session!.id)
      navigate('/session/complete', { state: { stars: starsEarned, total: words.length, childId } })
    } else {
      setItemIndex((i) => i + 1)
      setAttemptNum(1)
      setLastResult(null)
      setPhase('ready')
    }
  }

  if (phase === 'loading') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-sky-50">
        <p className="text-gray-500">Getting ready…</p>
      </div>
    )
  }

  if (phase === 'error') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-sky-50">
        <p className="text-red-600">{errorMessage}</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-sky-50 p-6">
      <p className="text-gray-500 mb-4">
        {child?.name} — Item {itemIndex + 1} of {words.length} — ⭐ {starsEarned}
      </p>

      <div className="bg-white rounded-2xl shadow-md p-10 w-full max-w-md">
        {phase === 'feedback' && lastResult ? (
          <FeedbackDisplay result={lastResult} attemptNum={attemptNum} onContinue={handleContinue} />
        ) : (
          currentWord && (
            <DrillWordCard
              word={currentWord.word}
              attemptNum={attemptNum}
              isRecording={phase === 'recording'}
              isProcessing={phase === 'processing'}
              onPlay={handlePlay}
              onRecord={handleRecordToggle}
            />
          )
        )}
      </div>
    </div>
  )
}
