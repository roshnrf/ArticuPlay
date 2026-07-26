interface Props {
  word: string
  attemptNum: number
  isRecording: boolean
  isProcessing: boolean
  onPlay: () => void
  onRecord: () => void
}

export function DrillWordCard({ word, attemptNum, isRecording, isProcessing, onPlay, onRecord }: Props) {
  return (
    <div className="text-center space-y-6">
      <p className="text-sm text-gray-500">Attempt {attemptNum} of 3</p>
      <p className="text-4xl font-bold text-sky-900">{word}</p>
      <button
        onClick={onPlay}
        disabled={isRecording || isProcessing}
        className="bg-white border-2 border-sky-300 rounded-full px-6 py-3 font-medium disabled:opacity-50"
      >
        🔊 Hear the word
      </button>
      <div>
        <button
          onClick={onRecord}
          disabled={isProcessing}
          className={`rounded-full px-8 py-4 font-semibold text-white text-lg disabled:opacity-50 ${
            isRecording ? 'bg-red-500 animate-pulse' : 'bg-sky-600'
          }`}
        >
          {isProcessing ? 'Checking…' : isRecording ? '⏹ Stop' : '🎤 Say it!'}
        </button>
      </div>
    </div>
  )
}
