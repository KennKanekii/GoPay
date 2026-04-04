import cors from 'cors'
import dotenv from 'dotenv'
import express from 'express'
import fs from 'node:fs'
import path from 'node:path'
import { createHash, randomBytes } from 'node:crypto'
import { fileURLToPath } from 'node:url'

// Load environment-specific config if present.
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const nodeEnv = process.env.NODE_ENV ?? 'development'
const envPath = path.join(__dirname, `.env.${nodeEnv}`)

if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath })
} else {
  dotenv.config({ path: path.join(__dirname, '.env') })
}

const app = express()

app.use(cors())
app.use(express.json())

type StoredUser = {
  id: string
  name: string
  identifier: string // normalized phone/email
  passwordSalt: string
  passwordHash: string
  createdAt: string
}

type SignupBody = {
  name?: unknown
  identifier?: unknown
  password?: unknown
}

type LoginBody = {
  identifier?: unknown
  password?: unknown
}

type Session = {
  token: string
  userId: string
  createdAt: string
}

const dataDir = path.join(__dirname, 'data')
const usersFilePath = path.join(dataDir, 'users.json')
const sessionsFilePath = path.join(dataDir, 'sessions.json')

function ensureUsersFile() {
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true })
  }
  if (!fs.existsSync(usersFilePath)) {
    fs.writeFileSync(usersFilePath, '[]', 'utf8')
  }
  if (!fs.existsSync(sessionsFilePath)) {
    fs.writeFileSync(sessionsFilePath, '[]', 'utf8')
  }
}

function readUsers(): StoredUser[] {
  ensureUsersFile()
  try {
    const raw = fs.readFileSync(usersFilePath, 'utf8')
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed as StoredUser[]
  } catch {
    return []
  }
}

function writeUsers(users: StoredUser[]) {
  ensureUsersFile()
  fs.writeFileSync(usersFilePath, JSON.stringify(users, null, 2), 'utf8')
}

function normalizeIdentifier(value: string) {
  return value.trim().toLowerCase()
}

function generateId() {
  return `user_${randomBytes(8).toString('hex')}`
}

function hashPassword(password: string, salt: string) {
  return createHash('sha256').update(`${salt}:${password}`).digest('hex')
}

function sendError(res: express.Response, status: number, message: string) {
  res.status(status).json({ ok: false, error: message })
}

function generateToken() {
  return `tok_${randomBytes(16).toString('hex')}`
}

function readSessions(): Session[] {
  ensureUsersFile()
  try {
    const raw = fs.readFileSync(sessionsFilePath, 'utf8')
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed as Session[]
  } catch {
    return []
  }
}

function writeSessions(sessions: Session[]) {
  ensureUsersFile()
  fs.writeFileSync(sessionsFilePath, JSON.stringify(sessions, null, 2), 'utf8')
}

function getAuthToken(req: express.Request) {
  const authHeader = req.headers.authorization
  if (typeof authHeader !== 'string') return null
  const match = /^Bearer\s+(.+)$/i.exec(authHeader)
  return match?.[1] ?? null
}

app.get('/health', (_req, res) => {
  res.json({ ok: true, service: 'gopay-backend' })
})

app.get('/api/v1/me', (req, res) => {
  const token = getAuthToken(req)
  if (!token) return sendError(res, 401, 'Not authenticated.')

  const sessions = readSessions()
  const session = sessions.find((s) => s.token === token)
  if (!session) return sendError(res, 401, 'Session expired.')

  const users = readUsers()
  const user = users.find((u) => u.id === session.userId)
  if (!user) return sendError(res, 401, 'Invalid user.')

  res.json({
    id: user.id,
    name: user.name,
    identifier: user.identifier,
  })
})

app.post('/api/v1/auth/signup', (req, res) => {
  const body = req.body as SignupBody | undefined

  const name = typeof body?.name === 'string' ? body.name.trim() : ''
  const identifierRaw = typeof body?.identifier === 'string' ? body.identifier : ''
  const password = typeof body?.password === 'string' ? body.password : ''

  if (name.length < 2) return sendError(res, 400, 'Name is required.')
  if (identifierRaw.trim().length < 3) return sendError(res, 400, 'Phone/email is required.')
  if (password.length < 6) return sendError(res, 400, 'Password must be at least 6 characters.')

  const identifier = normalizeIdentifier(identifierRaw)
  const users = readUsers()

  if (users.some((u) => u.identifier === identifier)) {
    return sendError(res, 409, 'An account with that phone/email already exists.')
  }

  const user: StoredUser = {
    id: generateId(),
    name,
    identifier,
    passwordSalt: randomBytes(16).toString('hex'),
    passwordHash: '',
    createdAt: new Date().toISOString(),
  }
  user.passwordHash = hashPassword(password, user.passwordSalt)

  users.push(user)
  writeUsers(users)

  res.status(201).json({ ok: true, id: user.id })
})

app.post('/api/v1/auth/login', (req, res) => {
  const body = req.body as LoginBody | undefined

  const identifierRaw = typeof body?.identifier === 'string' ? body.identifier : ''
  const password = typeof body?.password === 'string' ? body.password : ''

  if (identifierRaw.trim().length < 3) return sendError(res, 400, 'Phone/email is required.')
  if (password.length < 6) return sendError(res, 400, 'Invalid credentials.')

  const identifier = normalizeIdentifier(identifierRaw)
  const users = readUsers()
  const user = users.find((u) => u.identifier === identifier)
  if (!user) return sendError(res, 401, 'Invalid credentials.')

  const expectedHash = hashPassword(password, user.passwordSalt)
  if (expectedHash !== user.passwordHash) return sendError(res, 401, 'Invalid credentials.')

  const sessions = readSessions()
  const token = generateToken()
  sessions.push({ token, userId: user.id, createdAt: new Date().toISOString() })
  writeSessions(sessions)

  res.json({
    ok: true,
    token,
    user: {
      id: user.id,
      name: user.name,
      identifier: user.identifier,
    },
  })
})

app.post('/api/v1/auth/logout', (req, res) => {
  const token = getAuthToken(req)
  if (!token) return sendError(res, 401, 'Not authenticated.')

  const sessions = readSessions()
  const next = sessions.filter((s) => s.token !== token)
  writeSessions(next)

  res.json({ ok: true })
})

const port = Number(process.env.PORT ?? 8080)

// Optional: serve the built frontend (useful when backend is your "entrypoint").
const shouldServeFrontend =
  process.env.SERVE_FRONTEND === 'true' || nodeEnv === 'production'

if (shouldServeFrontend) {
  const frontendDist = path.resolve(__dirname, '../frontend/dist')
  if (fs.existsSync(frontendDist)) {
    app.use(express.static(frontendDist))
    app.get('*', (_req, res) => {
      res.sendFile(path.join(frontendDist, 'index.html'))
    })
  }
}

app.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`GoPay backend listening on http://localhost:${port}`)
})

