import { Link, useLocation } from 'react-router-dom'

interface SessionCompleteState {
  stars: number
  total: number
  childId: string
}

export function SessionComplete() {
  const location = useLocation()
  const state = location.state as SessionCompleteState | null

  return (
    <div className="min-h-screen flex items-center justify-center bg-sky-50 p-6">
      <div className="bg-white rounded-2xl shadow-md p-10 text-center max-w-sm w-full space-y-4">
        <p className="text-5xl">🎉</p>
        <h1 className="text-2xl font-bold text-sky-900">Session complete!</h1>
        {state && (
          <p className="text-lg text-gray-600">
            You earned <span className="font-bold text-amber-600">{state.stars}</span> out of{' '}
            {state.total} stars
          </p>
        )}
        <Link
          to="/children"
          className="inline-block bg-sky-600 text-white rounded-lg px-6 py-2 font-medium"
        >
          Back to children
        </Link>
      </div>
    </div>
  )
}
