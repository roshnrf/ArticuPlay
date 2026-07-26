import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider } from './auth/AuthContext'
import { ProtectedRoute } from './auth/ProtectedRoute'
import { Login } from './pages/Login'
import { Register } from './pages/Register'
import { Children } from './pages/Children'
import { NewChild } from './pages/NewChild'
import { Session } from './pages/Session'
import { SessionComplete } from './pages/SessionComplete'

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/children" replace />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/children"
            element={
              <ProtectedRoute>
                <Children />
              </ProtectedRoute>
            }
          />
          <Route
            path="/children/new"
            element={
              <ProtectedRoute>
                <NewChild />
              </ProtectedRoute>
            }
          />
          <Route
            path="/session/complete"
            element={
              <ProtectedRoute>
                <SessionComplete />
              </ProtectedRoute>
            }
          />
          <Route
            path="/session/:childId"
            element={
              <ProtectedRoute>
                <Session />
              </ProtectedRoute>
            }
          />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}

export default App
