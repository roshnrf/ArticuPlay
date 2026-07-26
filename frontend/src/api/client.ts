import type {
  ASRResult,
  ChildCreate,
  ChildRead,
  ScoreRequest,
  ScoreResult,
  SessionCreate,
  SessionRead,
  Token,
  UserRead,
  WordContentRead,
} from '../types/api'

const BASE_URL = import.meta.env.VITE_API_BASE_URL as string
const TOKEN_KEY = 'storyweaver_token'

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken()
  const headers = new Headers(options.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)

  const response = await fetch(`${BASE_URL}${path}`, { ...options, headers })
  if (!response.ok) {
    const detail = await response.text()
    throw new ApiError(response.status, detail || response.statusText)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

function jsonRequest<T>(path: string, method: string, body?: unknown): Promise<T> {
  const headers = new Headers({ 'Content-Type': 'application/json' })
  return request<T>(path, { method, headers, body: body ? JSON.stringify(body) : undefined })
}

// --- auth ---

export function register(email: string, password: string, full_name: string): Promise<UserRead> {
  return jsonRequest<UserRead>('/auth/register', 'POST', { email, password, full_name })
}

export async function login(email: string, password: string): Promise<Token> {
  const body = new URLSearchParams({ username: email, password })
  const token = await request<Token>('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })
  setToken(token.access_token)
  return token
}

// --- children ---

export function listChildren(): Promise<ChildRead[]> {
  return request<ChildRead[]>('/children/')
}

export function createChild(body: ChildCreate): Promise<ChildRead> {
  return jsonRequest<ChildRead>('/children/', 'POST', body)
}

export function getChild(childId: string): Promise<ChildRead> {
  return request<ChildRead>(`/children/${childId}`)
}

// --- content ---

export function getWords(level: number, lang = 'en'): Promise<WordContentRead[]> {
  return request<WordContentRead[]>(`/content/words/${level}?lang=${lang}`)
}

// --- session ---

export function startSession(body: SessionCreate): Promise<SessionRead> {
  return jsonRequest<SessionRead>('/session/start', 'POST', body)
}

export async function speakWord(word: string, language: string): Promise<Blob> {
  const response = await fetch(`${BASE_URL}/session/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ word, language }),
  })
  if (!response.ok) throw new ApiError(response.status, await response.text())
  return response.blob()
}

export async function transcribe(audioBlob: Blob, language: string, targetWord: string): Promise<ASRResult> {
  const formData = new FormData()
  formData.append('audio', audioBlob, 'attempt.webm')
  formData.append('language', language)
  formData.append('target_word', targetWord)

  const token = getToken()
  const headers = new Headers()
  if (token) headers.set('Authorization', `Bearer ${token}`)

  const response = await fetch(`${BASE_URL}/session/transcribe`, { method: 'POST', headers, body: formData })
  if (!response.ok) throw new ApiError(response.status, await response.text())
  return response.json()
}

export function scoreItem(body: ScoreRequest): Promise<ScoreResult> {
  return jsonRequest<ScoreResult>('/session/score', 'POST', body)
}

export function completeSession(sessionId: string): Promise<SessionRead> {
  return request<SessionRead>(`/session/complete?session_id=${sessionId}`, { method: 'POST' })
}
