# Demo Backend

FastAPI + WebSocket backend for the EEG CNN/LSTM demo UI.

## Setup
```bash
python -m venv .venv-demo
source .venv-demo/bin/activate
pip install -r demo_backend/requirements-demo-backend.txt
```

## Run
```bash
python demo_backend/server.py
```

## Endpoints
- `GET /health`
- `GET /schema`
- `POST /control` with JSON `{ "mode": "replay|live|idle", "replay_path": "...", "fps": 20, "device": "cpu|cuda", "mc_passes": 10 }`
- `WS /stream`

## Tests
```bash
python -m compileall demo_backend
pytest demo_backend/tests
```
