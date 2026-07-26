import type { ScoreResult } from '../types/api'
import { StarAnimation } from './StarAnimation'

interface Props {
  result: ScoreResult
  attemptNum: number
  onContinue: () => void
}

export function FeedbackDisplay({ result, attemptNum, onContinue }: Props) {
  const willAutoAdvance = !result.passed && attemptNum >= 3
  const firstError = result.errors[0]

  return (
    <div className="text-center space-y-4">
      {result.passed ? (
        <>
          <StarAnimation />
          <p className="text-xl font-semibold text-green-700">Great job!</p>
        </>
      ) : (
        <>
          <p className="text-xl font-semibold text-amber-700">Almost!</p>
          {firstError && (
            <p className="text-gray-600">
              The <span className="font-mono font-bold">/{firstError.expected}/</span> sound needs a
              little work.
            </p>
          )}
          {willAutoAdvance && <p className="text-sm text-gray-500">Let's try the next word.</p>}
        </>
      )}
      <button
        onClick={onContinue}
        className="bg-sky-600 text-white rounded-lg px-6 py-2 font-medium"
      >
        {result.passed || willAutoAdvance ? 'Next word' : 'Try again'}
      </button>
    </div>
  )
}
