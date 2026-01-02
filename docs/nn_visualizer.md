# NN Visualizer

The NN Visualizer page adds an interactive view of the CNN+LSTM model used by the EEG demo. It can show architecture, weights, activations, and (optionally) a timeline of checkpoint snapshots.

## Features
- Architecture graph with shapes, params, and MACs per layer
- Weights heatmaps for Conv1d kernels, Linear heads, and LSTM matrices
- Activations for a selected window (online websocket stream or offline npz replay)
- Optional timeline scrubber if `exports/nnvis_timeline/manifest.json` exists

## Endpoints
- `GET /nnvis/manifest` for architecture + labels
- `GET /nnvis/weights` for current weights
- `GET /nnvis/offline/sources` for available npz files
- `GET /nnvis/offline/sample` for activation payloads by index
- `GET /nnvis/timeline/manifest` and `GET /nnvis/timeline/weights` if timeline exports exist

## Timeline Export
Use the helper script to convert checkpoint files into a timeline bundle:

```
python scripts/export_nnvis_timeline.py --checkpoints checkpoints --pattern "*.pt"
```

This creates `exports/nnvis_timeline/manifest.json` and weights files that the UI can scrub.
