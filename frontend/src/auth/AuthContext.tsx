import { createContext, useContext, useState, type ReactNode } from 'react'
import { clearToken, getToken, login as apiLogin, register as apiRegister } from '../api/client'

interface AuthContextValue {
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, fullName: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(() => getToken() !== null)

  async function login(email: string, password: string) {
    await apiLogin(email, password)
    setIsAuthenticated(true)
  }

  async function register(email: string, password: string, fullName: string) {
    await apiRegister(email, password, fullName)
    await login(email, password)
  }

  function logout() {
    clearToken()
    setIsAuthenticated(false)
  }

  return (
    <AuthContext.Provider value={{ isAuthenticated, login, register, logout }}>{children}</AuthContext.Provider>
  )
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
