import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { listChildren } from '../api/client'
import type { ChildRead } from '../types/api'
import { useAuth } from '../auth/AuthContext'

export function Children() {
  const [children, setChildren] = useState<ChildRead[]>([])
  const [loading, setLoading] = useState(true)
  const { logout } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    listChildren()
      .then(setChildren)
      .finally(() => setLoading(false))
  }, [])

  function handleLogout() {
    logout()
    navigate('/login')
  }

  return (
    <div className="min-h-screen bg-sky-50 p-6">
      <div className="max-w-2xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-sky-900">Your children</h1>
          <button onClick={handleLogout} className="text-sm text-gray-500 hover:text-gray-700">
            Log out
          </button>
        </div>

        {loading && <p className="text-gray-500">Loading…</p>}

        {!loading && children.length === 0 && (
          <p className="text-gray-500 mb-4">No children added yet.</p>
        )}

        <div className="space-y-3 mb-6">
          {children.map((child) => (
            <Link
              key={child.id}
              to={`/session/${child.id}`}
              className="block bg-white rounded-xl p-4 shadow-sm hover:shadow-md transition"
            >
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-semibold text-lg">{child.name}</p>
                  <p className="text-sm text-gray-500">
                    Age {child.age} · Level {child.current_level} · {child.language.toUpperCase()}
                  </p>
                </div>
                <p className="text-sky-600 font-medium">Start session →</p>
              </div>
            </Link>
          ))}
        </div>

        <Link
          to="/children/new"
          className="inline-block bg-sky-600 text-white rounded-lg px-4 py-2 font-medium"
        >
          + Add a child
        </Link>
      </div>
    </div>
  )
}
