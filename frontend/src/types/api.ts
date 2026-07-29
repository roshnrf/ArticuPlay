// Mirrors backend/app/schemas/*.py exactly — keep in sync by hand, there's no
// codegen step, so field name/shape changes on either side need a manual update here.

export interface UserRead {
  id: string
  email: string
  full_name: string | null
  role: string
  is_active: boolean
}

export interface Token {
  access_token: string
  token_type: string
}

export interface ChildCreate {
  name: string
  age: number
  language: string
  current_level: number
  avatar_url?: string | null
}

export interface ChildRead {
  id: string
  parent_id: string
  name: string
  age: number
  language: string
  current_level: number
  avatar_url: string | null
  total_sessions: number
  streak_days: number
  badges: string
}

export interface SessionCreate {
  child_id: string
  language: string
  level: number
}

export interface SessionRead {
  id: string
  child_id: string
  language: string
  level: number
  status: string
  items_count: number
  correct_count: number
  started_at: string
  completed_at: string | null
}

export interface WordContentRead {
  id: string
  language: string
  level: number
  word: string
  ipa: string
  image_url: string | null
  audio_url: string | null
}

export interface ScoreRequest {
  session_id: string
  item_index: number
  target_word: string
  language: string
  child_transcript: string
  attempt_num: number
  phone_classifier_flag: boolean | null
}

export interface PhonemeErrorRead {
  position: number
  expected: string | null
  got: string | null
  type: string
}

export interface ScoreResult {
  item_index: number
  accuracy_score: number
  errors: PhonemeErrorRead[]
  passed: boolean
  attempt_num: number
  phone_classifier_override: boolean
}

export interface ASRResult {
  transcript: string
  language: string
  latency_sec: number
  words: { word: string; start: number; end: number }[] | null
  phone_classifier_flag: boolean | null
}
