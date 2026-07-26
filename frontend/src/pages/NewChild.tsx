import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { createChild } from '../api/client'

export function NewChild() {
  const [name, setName] = useState('')
  const [age, setAge] = useState(6)
  const [language, setLanguage] = useState('en')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await createChild({ name, age, language, current_level: 1 })
      navigate('/children')
    } catch {
      setError('Could not create child profile. Age must be between 3 and 12.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-sky-50">
      <form onSubmit={handleSubmit} className="bg-white p-8 rounded-2xl shadow-md w-full max-w-sm space-y-4">
        <h1 className="text-2xl font-bold text-center text-sky-900">Add a child</h1>
        <input
          type="text"
          placeholder="Child's name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
          className="w-full border rounded-lg px-3 py-2"
        />
        <div>
          <label className="text-sm text-gray-600">Age</label>
          <input
            type="number"
            min={3}
            max={12}
            value={age}
            onChange={(e) => setAge(Number(e.target.value))}
            required
            className="w-full border rounded-lg px-3 py-2"
          />
        </div>
        <div>
          <label className="text-sm text-gray-600">Language</label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="w-full border rounded-lg px-3 py-2"
          >
            <option value="en">English</option>
            <option value="ar">Arabic</option>
            <option value="hi">Hindi</option>
            <option value="zh">Mandarin</option>
          </select>
        </div>
        {error && <p className="text-red-600 text-sm">{error}</p>}
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-sky-600 text-white rounded-lg py-2 font-medium disabled:opacity-50"
        >
          {loading ? 'Creating…' : 'Add child'}
        </button>
      </form>
    </div>
  )
}
