# EEG Demo App (Tauri + React)

Polished UI for live/replay inference and diagnostics.

## Setup
```bash
cd demo_app
npm install
```

## Run (web)
```bash
npm run dev
```

## Run (desktop)
```bash
npm run tauri:dev
```

## Tests
```bash
npm run typecheck
npm run test
npm run build
```

## Backend
Make sure the backend is running on `http://127.0.0.1:8008`.

```bash
python demo_backend/server.py
```
