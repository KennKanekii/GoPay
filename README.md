# GoPay (Frontend + Backend)

This workspace contains:

- `frontend/`: **GoPay** web UI (Vite + React + TypeScript + Tailwind)
- `backend/`: Node **API service** (Express + TypeScript)

## Run locally

### Frontend

```bash
cd frontend
npm install
npm start
```

### Backend

```bash
cd backend
npm install
npm run dev
```

Note: `server.tsx` is TypeScript, so to run it directly with Node you need a TS loader. The backend uses `tsx` under the hood (via `node --import tsx ...`).

Backend can also serve the built frontend (from `frontend/dist`) in production if you set:

```bash
SERVE_FRONTEND=true
```

If you really want a single command, use:

```bash
cd backend
node --import tsx server.tsx
```

## Production build

```bash
cd frontend
npm run build
npm run preview
```

