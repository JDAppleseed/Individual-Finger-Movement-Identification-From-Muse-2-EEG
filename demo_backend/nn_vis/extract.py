from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from demo_backend.utils_demo import apply_channel_normalizer
from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.label_schema import ACTION_NAMES, FINGER_NAMES


@dataclass
class PackedArrayPayload:
    data: object
    shape: List[int]


def _device_of(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def load_model_and_weights(
    model_path: str, device: str = "cpu"
) -> CNNLSTMFingerActionNet:
    state = torch.load(model_path, map_location="cpu")
    n_fingers = int(state["finger_head.weight"].shape[0])
    n_actions = int(state["action_head.weight"].shape[0])
    model = CNNLSTMFingerActionNet(
        n_channels=4, n_fingers=n_fingers, n_actions=n_actions
    )
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    return model


def _count_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _conv_macs(out_ch: int, out_t: int, in_ch: int, k: int) -> int:
    return int(out_ch * out_t * in_ch * k)


def _linear_macs(in_features: int, out_features: int) -> int:
    return int(in_features * out_features)


def _lstm_macs(input_size: int, hidden_size: int, timesteps: int) -> int:
    per_timestep = 4 * (input_size * hidden_size + hidden_size * hidden_size)
    return int(per_timestep * timesteps)


def _pack_array(
    array: np.ndarray,
    *,
    quantize: bool = True,
    max_abs: Optional[float] = None,
    downsample: int = 1,
    min_raw_size: int = 4096,
) -> object:
    arr = np.asarray(array)
    if max_abs is not None:
        arr = np.clip(arr, -max_abs, max_abs)
    if downsample > 1:
        if arr.ndim == 1:
            arr = arr[::downsample]
        elif arr.ndim == 2:
            arr = arr[::downsample, ::downsample]
        elif arr.ndim >= 3:
            slices = [slice(None)] * arr.ndim
            slices[0] = slice(None, None, downsample)
            slices[1] = slice(None, None, downsample)
            arr = arr[tuple(slices)]
    if arr.size <= min_raw_size and not quantize:
        return arr.tolist()
    if arr.size <= min_raw_size:
        return arr.astype(np.float32).tolist()

    if quantize:
        packed = arr.astype(np.float16)
        raw = packed.tobytes()
        return {
            "encoding": "f16_base64",
            "shape": list(packed.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }

    return arr.astype(np.float32).tolist()


def pack_tensor(
    array: np.ndarray, *, quantize: bool = True, min_raw_size: int = 4096
) -> object:
    """Pack activations to reduce websocket payload size."""
    arr = np.asarray(array)
    if arr.size < min_raw_size:
        return arr.astype(np.float32).tolist()
    if quantize:
        packed = arr.astype(np.float16)
        raw = packed.tobytes()
        return {
            "encoding": "f16_base64",
            "shape": list(packed.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    return arr.astype(np.float32).tolist()


def _topk_edges(matrix: np.ndarray, k: int, name: str) -> List[Dict[str, object]]:
    flat = np.abs(matrix).ravel()
    if flat.size == 0:
        return []
    k = int(min(k, flat.size))
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(-flat[idx])]
    rows, cols = np.unravel_index(idx, matrix.shape)
    edges = []
    for r, c in zip(rows, cols):
        edges.append(
            {
                "matrix": name,
                "i": int(r),
                "j": int(c),
                "v": float(matrix[r, c]),
            }
        )
    return edges


def extract_architecture_manifest(
    model: CNNLSTMFingerActionNet, timeline_available: bool
) -> Dict[str, object]:
    conv1 = model.conv[0]
    gn1 = model.conv[1]
    conv2 = model.conv[4]
    gn2 = model.conv[5]

    conv1_params = _count_params(conv1)
    conv2_params = _count_params(conv2)
    gn1_params = _count_params(gn1)
    gn2_params = _count_params(gn2)
    lstm_params = _count_params(model.lstm)
    finger_params = _count_params(model.finger_head)
    action_params = _count_params(model.action_head)

    conv1_macs = _conv_macs(16, 64, 4, 5)
    conv2_macs = _conv_macs(32, 64, 16, 3)
    lstm_macs = _lstm_macs(32, 64, 64)
    finger_macs = _linear_macs(64, 6)
    action_macs = _linear_macs(64, 3)

    totals = (
        conv1_params
        + conv2_params
        + gn1_params
        + gn2_params
        + lstm_params
        + finger_params
        + action_params
    )
    macs_total = conv1_macs + conv2_macs + lstm_macs + finger_macs + action_macs

    nodes = [
        {
            "id": "input",
            "title": "Input",
            "kind": "input",
            "shape": "[64,4]",
            "params": 0,
            "macs": 0,
        },
        {
            "id": "conv1",
            "title": "Conv1d 4→16 k=5",
            "kind": "conv1d",
            "shape_in": "[4,64]",
            "shape_out": "[16,64]",
            "params": conv1_params,
            "macs": conv1_macs,
        },
        {
            "id": "gn1",
            "title": "GroupNorm",
            "kind": "norm",
            "shape": "[16,64]",
            "params": gn1_params,
            "macs": 0,
        },
        {
            "id": "relu1",
            "title": "ReLU",
            "kind": "activation",
            "shape": "[16,64]",
            "params": 0,
            "macs": 0,
        },
        {
            "id": "drop1",
            "title": "Dropout",
            "kind": "dropout",
            "shape": "[16,64]",
            "params": 0,
            "macs": 0,
        },
        {
            "id": "conv2",
            "title": "Conv1d 16→32 k=3",
            "kind": "conv1d",
            "shape_in": "[16,64]",
            "shape_out": "[32,64]",
            "params": conv2_params,
            "macs": conv2_macs,
        },
        {
            "id": "gn2",
            "title": "GroupNorm",
            "kind": "norm",
            "shape": "[32,64]",
            "params": gn2_params,
            "macs": 0,
        },
        {
            "id": "relu2",
            "title": "ReLU",
            "kind": "activation",
            "shape": "[32,64]",
            "params": 0,
            "macs": 0,
        },
        {
            "id": "drop2",
            "title": "Dropout",
            "kind": "dropout",
            "shape": "[32,64]",
            "params": 0,
            "macs": 0,
        },
        {
            "id": "lstm",
            "title": "LSTM 32→64",
            "kind": "lstm",
            "shape_in": "[64,32]",
            "shape_out": "[64,64]",
            "params": lstm_params,
            "macs": lstm_macs,
        },
        {
            "id": "last",
            "title": "Last Timestep",
            "kind": "pool",
            "shape": "[64]",
            "params": 0,
            "macs": 0,
        },
        {
            "id": "head_dropout",
            "title": "Head Dropout",
            "kind": "dropout",
            "shape": "[64]",
            "params": 0,
            "macs": 0,
        },
        {
            "id": "finger_head",
            "title": "Finger Head 64→6",
            "kind": "linear",
            "shape_in": "[64]",
            "shape_out": "[6]",
            "params": finger_params,
            "macs": finger_macs,
        },
        {
            "id": "action_head",
            "title": "Action Head 64→3",
            "kind": "linear",
            "shape_in": "[64]",
            "shape_out": "[3]",
            "params": action_params,
            "macs": action_macs,
        },
    ]

    edges = [
        {"from": "input", "to": "conv1"},
        {"from": "conv1", "to": "gn1"},
        {"from": "gn1", "to": "relu1"},
        {"from": "relu1", "to": "drop1"},
        {"from": "drop1", "to": "conv2"},
        {"from": "conv2", "to": "gn2"},
        {"from": "gn2", "to": "relu2"},
        {"from": "relu2", "to": "drop2"},
        {"from": "drop2", "to": "lstm"},
        {"from": "lstm", "to": "last"},
        {"from": "last", "to": "head_dropout"},
        {"from": "head_dropout", "to": "finger_head"},
        {"from": "head_dropout", "to": "action_head"},
    ]

    timeline: Dict[str, object] = {"available": timeline_available}
    if timeline_available:
        timeline["manifest_url"] = "/nnvis/timeline/manifest"

    return {
        "model_name": "CNNLSTMFingerActionNet",
        "input": {
            "timesteps": 64,
            "channels": 4,
            "channel_names": ["TP9", "AF7", "AF8", "TP10"],
        },
        "labels": {
            "action": {str(k): v for k, v in ACTION_NAMES.items()},
            "finger": {str(k): v for k, v in FINGER_NAMES.items()},
        },
        "nodes": nodes,
        "edges": edges,
        "totals": {"params": totals, "macs_per_window": macs_total},
        "timeline": timeline,
    }


def extract_weights(
    model: CNNLSTMFingerActionNet,
    *,
    quantize: bool = True,
    max_abs: Optional[float] = None,
    downsample: int = 1,
    topk: int = 150,
) -> Dict[str, object]:
    conv1 = model.conv[0]
    conv2 = model.conv[4]

    weights = {
        "version": 1,
        "conv": [
            {
                "id": "conv1",
                "weight_shape": list(conv1.weight.shape),
                "weights": _pack_array(
                    conv1.weight.detach().cpu().numpy(),
                    quantize=quantize,
                    max_abs=max_abs,
                    downsample=downsample,
                ),
                "bias": _pack_array(
                    conv1.bias.detach().cpu().numpy(),
                    quantize=quantize,
                    max_abs=max_abs,
                    downsample=downsample,
                )
                if conv1.bias is not None
                else None,
            },
            {
                "id": "conv2",
                "weight_shape": list(conv2.weight.shape),
                "weights": _pack_array(
                    conv2.weight.detach().cpu().numpy(),
                    quantize=quantize,
                    max_abs=max_abs,
                    downsample=downsample,
                ),
                "bias": _pack_array(
                    conv2.bias.detach().cpu().numpy(),
                    quantize=quantize,
                    max_abs=max_abs,
                    downsample=downsample,
                )
                if conv2.bias is not None
                else None,
            },
        ],
        "linear": [
            {
                "id": "finger_head",
                "weight_shape": list(model.finger_head.weight.shape),
                "weights": _pack_array(
                    model.finger_head.weight.detach().cpu().numpy(),
                    quantize=quantize,
                    max_abs=max_abs,
                    downsample=downsample,
                ),
                "bias": _pack_array(
                    model.finger_head.bias.detach().cpu().numpy(),
                    quantize=quantize,
                    max_abs=max_abs,
                ),
            },
            {
                "id": "action_head",
                "weight_shape": list(model.action_head.weight.shape),
                "weights": _pack_array(
                    model.action_head.weight.detach().cpu().numpy(),
                    quantize=quantize,
                    max_abs=max_abs,
                    downsample=downsample,
                ),
                "bias": _pack_array(
                    model.action_head.bias.detach().cpu().numpy(),
                    quantize=quantize,
                    max_abs=max_abs,
                ),
            },
        ],
    }

    weight_ih = model.lstm.weight_ih_l0.detach().cpu().numpy()
    weight_hh = model.lstm.weight_hh_l0.detach().cpu().numpy()
    bias_ih = model.lstm.bias_ih_l0.detach().cpu().numpy()
    bias_hh = model.lstm.bias_hh_l0.detach().cpu().numpy()

    top_edges = _topk_edges(weight_ih, topk, "weight_ih_l0") + _topk_edges(
        weight_hh, topk, "weight_hh_l0"
    )

    weights["lstm"] = {
        "id": "lstm",
        "weight_ih_l0_shape": list(weight_ih.shape),
        "weight_hh_l0_shape": list(weight_hh.shape),
        "bias_ih_l0_shape": list(bias_ih.shape),
        "bias_hh_l0_shape": list(bias_hh.shape),
        "weight_ih_l0": _pack_array(
            weight_ih, quantize=quantize, max_abs=max_abs, downsample=downsample
        ),
        "weight_hh_l0": _pack_array(
            weight_hh, quantize=quantize, max_abs=max_abs, downsample=downsample
        ),
        "bias_ih_l0": _pack_array(bias_ih, quantize=quantize, max_abs=max_abs),
        "bias_hh_l0": _pack_array(bias_hh, quantize=quantize, max_abs=max_abs),
        "topk": {"k": int(topk), "edges": top_edges},
    }

    return weights


def extract_activations(
    model: CNNLSTMFingerActionNet,
    window_TxC: np.ndarray,
    *,
    deterministic: bool = True,
    normalizer: Optional[object] = None,
    mc_passes: Optional[int] = None,
) -> Tuple[Dict[str, object], Dict[str, float | bool | None]]:
    device = _device_of(model)
    window_TxC = window_TxC.astype(np.float32)
    window_TxC = apply_channel_normalizer(window_TxC, normalizer)

    x = torch.tensor(window_TxC, dtype=torch.float32, device=device).unsqueeze(0)

    was_training = model.training
    if deterministic:
        model.eval()

    x_perm = x.permute(0, 2, 1)
    conv1 = model.conv[0](x_perm)
    conv1 = model.conv[1](conv1)
    conv1 = model.conv[2](conv1)
    conv1 = model.conv[3](conv1)

    conv2 = model.conv[4](conv1)
    conv2 = model.conv[5](conv2)
    conv2 = model.conv[6](conv2)
    conv2 = model.conv[7](conv2)

    lstm_in = conv2.permute(0, 2, 1)
    lstm_out, _ = model.lstm(lstm_in)
    last = lstm_out[:, -1, :]
    features = model.head_dropout(last)

    finger_logits = model.finger_head(features)
    action_logits = model.action_head(features)
    finger_probs = F.softmax(finger_logits, dim=-1)
    action_probs = F.softmax(action_logits, dim=-1)

    if deterministic and was_training:
        model.train()

    activations = {
        "input": window_TxC,
        "conv1": conv1.squeeze(0).detach().cpu().numpy(),
        "conv2": conv2.squeeze(0).detach().cpu().numpy(),
        "lstm_out": lstm_out.squeeze(0).detach().cpu().numpy(),
        "last_features": features.squeeze(0).detach().cpu().numpy(),
        "finger_probs": finger_probs.squeeze(0).detach().cpu().numpy(),
        "action_probs": action_probs.squeeze(0).detach().cpu().numpy(),
    }

    uncertainty: Dict[str, float | bool | None] = {
        "present": False,
        "finger_std_mean": None,
        "action_std_mean": None,
        "finger_entropy": None,
        "action_entropy": None,
        "finger_mi": None,
        "action_mi": None,
    }

    if mc_passes is not None and mc_passes > 0 and hasattr(model, "mc_forward"):
        mc = model.mc_forward(x, passes=mc_passes)
        finger_std = mc["finger_std"].squeeze(0).detach().cpu().numpy()
        action_std = mc["action_std"].squeeze(0).detach().cpu().numpy()
        uncertainty = {
            "present": True,
            "finger_std_mean": float(np.mean(finger_std)),
            "action_std_mean": float(np.mean(action_std)),
            "finger_entropy": float(mc["finger_entropy"].item()),
            "action_entropy": float(mc["action_entropy"].item()),
            "finger_mi": float(mc["finger_mi"].item()),
            "action_mi": float(mc["action_mi"].item()),
        }

    return activations, uncertainty
