#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.label_schema import ACTION_NAMES, ACTION_REST, FINGER_NAMES, event_type_for
from utils.sequence_data import load_sequence_npz
from visualization.muse_topomap import (
    compute_bandpower_windows,
    compute_map_limits,
    mean_bandpower_map,
    plot_muse_topomap_grid,
    split_indices_in_halves,
)


METRIC_CONFIG = {
    "absolute": {
        "cmap": "jet",
        "center_zero": False,
        "colorbar_label": "Band Power",
        "title_suffix": "Absolute Power",
    },
    "log_absolute": {
        "cmap": "jet",
        "center_zero": False,
        "colorbar_label": "log10 Band Power",
        "title_suffix": "Log Power",
    },
    "rest_delta": {
        "cmap": "jet",
        "center_zero": True,
        "colorbar_label": "Delta vs REST (log10 power)",
        "title_suffix": "Delta vs REST",
    },
    "rest_zscore": {
        "cmap": "jet",
        "center_zero": True,
        "colorbar_label": "Z vs REST (log-power)",
        "title_suffix": "Z vs REST",
    },
}


class FigureSpec(NamedTuple):
    filename: str
    group_by: str
    metric: str
    include_none: bool = False
    split_halves: bool = False


def _scalar_float(value, default: float) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _channel_names_from_meta(meta) -> np.ndarray:
    channel_names = meta.get("channel_names")
    if channel_names is None:
        raise KeyError("NPZ is missing channel_names metadata")
    return np.asarray(channel_names).astype("U").reshape(-1)


def _safe_log_bandpower(bandpower: np.ndarray) -> np.ndarray:
    arr = np.asarray(bandpower, dtype=np.float32)
    return np.log10(np.maximum(arr, 1e-6)).astype(np.float32)


def _rest_reference(log_bandpower: np.ndarray, y_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rest_idx = np.asarray(y_action, dtype=np.int64) == int(ACTION_REST)
    if not np.any(rest_idx):
        raise ValueError("REST windows are required for rest-relative metrics")
    rest = np.asarray(log_bandpower[rest_idx], dtype=np.float32)
    rest_mean = rest.mean(axis=0).astype(np.float32)
    rest_std = rest.std(axis=0).astype(np.float32)
    rest_std = np.where(rest_std < 1e-6, 1.0, rest_std).astype(np.float32)
    return rest_mean, rest_std


def transform_bandpower(
    bandpower: np.ndarray,
    y_action: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric == "absolute":
        return np.asarray(bandpower, dtype=np.float32)
    log_bandpower = _safe_log_bandpower(bandpower)
    if metric == "log_absolute":
        return log_bandpower
    rest_mean, rest_std = _rest_reference(log_bandpower, y_action)
    if metric == "rest_delta":
        return (log_bandpower - rest_mean).astype(np.float32)
    if metric == "rest_zscore":
        return ((log_bandpower - rest_mean) / rest_std).astype(np.float32)
    raise ValueError(f"Unsupported metric={metric}")


def _action_maps(values_windows, y_action):
    maps = []
    for action_id in sorted(ACTION_NAMES):
        idx = np.flatnonzero(y_action == action_id)
        if idx.size == 0:
            continue
        maps.append((ACTION_NAMES[action_id], mean_bandpower_map(values_windows, idx)))
    return maps


def _finger_maps(values_windows, y_finger, include_none: bool):
    maps = []
    for finger_id in sorted(FINGER_NAMES):
        if finger_id == 0 and not include_none:
            continue
        idx = np.flatnonzero(y_finger == finger_id)
        if idx.size == 0:
            continue
        maps.append((FINGER_NAMES[finger_id], mean_bandpower_map(values_windows, idx)))
    return maps


def _joint_label(action_id: int, finger_id: int) -> str:
    label = event_type_for(int(action_id), int(finger_id))
    if int(action_id) == ACTION_REST:
        return "REST"
    return label.replace("_", " ").upper()


def _joint_maps(values_windows, y_action, y_finger):
    ordered_pairs = [(ACTION_REST, 0)]
    for action_id in sorted(ACTION_NAMES):
        if action_id == ACTION_REST:
            continue
        for finger_id in sorted(FINGER_NAMES):
            if finger_id == 0:
                continue
            ordered_pairs.append((action_id, finger_id))

    maps = []
    for action_id, finger_id in ordered_pairs:
        idx = np.flatnonzero((y_action == action_id) & (y_finger == finger_id))
        if idx.size == 0:
            continue
        maps.append((_joint_label(action_id, finger_id), mean_bandpower_map(values_windows, idx)))
    return maps


def _split_half_action_maps(values_windows, y_action):
    maps = []
    for row_name, half_idx in (("Early", 0), ("Late", 1)):
        for action_id in sorted(ACTION_NAMES):
            idx = np.flatnonzero(y_action == action_id)
            if idx.size == 0:
                continue
            first, second = split_indices_in_halves(idx)
            chosen = (first, second)[half_idx]
            maps.append((f"{row_name} {ACTION_NAMES[action_id]}", mean_bandpower_map(values_windows, chosen)))
    return maps


def build_maps(
    values_windows,
    y_action,
    y_finger,
    *,
    group_by: str,
    include_none: bool,
    split_halves: bool,
):
    if split_halves:
        if group_by != "action":
            raise ValueError("--split-halves currently supports only --group-by action")
        return _split_half_action_maps(values_windows, y_action), 3
    if group_by == "action":
        return _action_maps(values_windows, y_action), 3
    if group_by == "finger":
        return _finger_maps(values_windows, y_finger, include_none=include_none), 3
    if group_by == "joint":
        return _joint_maps(values_windows, y_action, y_finger), 4
    raise ValueError(f"Unsupported group_by={group_by}")


def _band_slug(band: tuple[float, float]) -> str:
    if abs(band[0] - 8.0) < 1e-6 and abs(band[1] - 12.0) < 1e-6:
        return "alpha"
    low = str(band[0]).replace(".", "p")
    high = str(band[1]).replace(".", "p")
    return f"{low}_{high}hz"


def _metric_title(metric: str) -> str:
    return str(METRIC_CONFIG[metric]["title_suffix"])


def make_figure_title(
    *,
    group_by: str,
    metric: str,
    band: tuple[float, float],
    split_halves: bool,
    include_none: bool,
) -> str:
    title = f"Muse 2 {group_by.title()} Topomaps ({band[0]:.1f}-{band[1]:.1f} Hz) | {_metric_title(metric)}"
    if split_halves:
        title = f"{title} | Early vs Late"
    if group_by == "finger" and include_none:
        title = f"{title} | Includes NONE"
    return title


def default_suite_specs(band: tuple[float, float]) -> list[FigureSpec]:
    band_slug = _band_slug(band)
    return [
        FigureSpec(
            filename=f"experimental_muse_action_{band_slug}_log_absolute_topomaps.png",
            group_by="action",
            metric="log_absolute",
        ),
        FigureSpec(
            filename=f"experimental_muse_action_{band_slug}_rest_delta_topomaps.png",
            group_by="action",
            metric="rest_delta",
        ),
        FigureSpec(
            filename=f"experimental_muse_action_{band_slug}_rest_zscore_topomaps.png",
            group_by="action",
            metric="rest_zscore",
        ),
        FigureSpec(
            filename=f"experimental_muse_action_{band_slug}_split_halves_log_absolute_topomaps.png",
            group_by="action",
            metric="log_absolute",
            split_halves=True,
        ),
        FigureSpec(
            filename=f"experimental_muse_action_{band_slug}_split_halves_rest_zscore_topomaps.png",
            group_by="action",
            metric="rest_zscore",
            split_halves=True,
        ),
        FigureSpec(
            filename=f"experimental_muse_finger_{band_slug}_rest_delta_topomaps.png",
            group_by="finger",
            metric="rest_delta",
        ),
        FigureSpec(
            filename=f"experimental_muse_finger_{band_slug}_rest_zscore_topomaps.png",
            group_by="finger",
            metric="rest_zscore",
        ),
        FigureSpec(
            filename=f"experimental_muse_finger_{band_slug}_with_none_rest_delta_topomaps.png",
            group_by="finger",
            metric="rest_delta",
            include_none=True,
        ),
        FigureSpec(
            filename=f"experimental_muse_joint_{band_slug}_rest_delta_topomaps.png",
            group_by="joint",
            metric="rest_delta",
        ),
    ]


def _round_dict(channel_names: np.ndarray, values: np.ndarray, digits: int = 4) -> dict[str, float]:
    return {
        str(name): round(float(val), digits)
        for name, val in zip(channel_names.tolist(), np.asarray(values, dtype=float).tolist())
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _pairwise_group_stats(
    bandpower: np.ndarray,
    y_action: np.ndarray,
    y_finger: np.ndarray,
    channel_names: np.ndarray,
) -> dict[str, dict]:
    log_bandpower = _safe_log_bandpower(bandpower)
    rest_mean_log, rest_std_log = _rest_reference(log_bandpower, y_action)
    rest_delta_windows = log_bandpower - rest_mean_log
    rest_z_windows = rest_delta_windows / rest_std_log

    action_stats = {}
    for action_id, action_name in ACTION_NAMES.items():
        idx = np.flatnonzero(y_action == action_id)
        if idx.size == 0:
            continue
        action_stats[action_name] = {
            "n": int(idx.size),
            "raw_mean": _round_dict(channel_names, mean_bandpower_map(bandpower, idx)),
            "log_mean": _round_dict(channel_names, mean_bandpower_map(log_bandpower, idx)),
            "delta_vs_rest_log": _round_dict(channel_names, mean_bandpower_map(rest_delta_windows, idx)),
            "z_vs_rest": _round_dict(channel_names, mean_bandpower_map(rest_z_windows, idx)),
        }

    finger_stats = {}
    for finger_id, finger_name in FINGER_NAMES.items():
        idx = np.flatnonzero(y_finger == finger_id)
        if idx.size == 0:
            continue
        finger_stats[finger_name] = {
            "n": int(idx.size),
            "raw_mean": _round_dict(channel_names, mean_bandpower_map(bandpower, idx)),
            "delta_vs_rest_log": _round_dict(channel_names, mean_bandpower_map(rest_delta_windows, idx)),
            "z_vs_rest": _round_dict(channel_names, mean_bandpower_map(rest_z_windows, idx)),
        }

    split_halves = {}
    for action_id, action_name in ACTION_NAMES.items():
        idx = np.flatnonzero(y_action == action_id)
        if idx.size == 0:
            continue
        first, second = split_indices_in_halves(idx)
        early_raw = mean_bandpower_map(bandpower, first)
        late_raw = mean_bandpower_map(bandpower, second)
        ratio = late_raw / np.maximum(early_raw, 1e-6)
        split_halves[action_name] = {
            "early_raw_mean": _round_dict(channel_names, early_raw),
            "late_raw_mean": _round_dict(channel_names, late_raw),
            "late_over_early": _round_dict(channel_names, ratio),
        }

    return {
        "rest_raw_mean": _round_dict(channel_names, mean_bandpower_map(bandpower, np.flatnonzero(y_action == ACTION_REST))),
        "rest_raw_std": _round_dict(channel_names, np.asarray(bandpower[y_action == ACTION_REST], dtype=np.float32).std(axis=0)),
        "rest_log_mean": _round_dict(channel_names, rest_mean_log),
        "rest_log_std": _round_dict(channel_names, rest_std_log),
        "action_stats": action_stats,
        "finger_stats": finger_stats,
        "split_halves": split_halves,
    }


def _interpret_results(summary: dict, channel_names: np.ndarray) -> list[str]:
    channels = channel_names.tolist()
    rest_raw = np.array([summary["rest_raw_mean"][c] for c in channels], dtype=np.float32)
    sorted_idx = np.argsort(rest_raw)[::-1]
    dominant_channel = channels[int(sorted_idx[0])]
    second_channel = channels[int(sorted_idx[1])] if len(channels) > 1 else channels[int(sorted_idx[0])]
    dominant_ratio = float(rest_raw[sorted_idx[0]] / max(rest_raw[sorted_idx[1]], 1e-6))

    lines = []
    if dominant_ratio >= 2.0:
        lines.append(
            f"REST alpha power is strongly dominated by {dominant_channel} versus {second_channel} "
            f"(about {dominant_ratio:.2f}x larger), so absolute topomaps mostly reflect channel scale."
        )
    else:
        lines.append("REST alpha power is not dominated by a single channel, so absolute maps are less distorted by baseline scale.")

    split_rest = summary["split_halves"].get("REST")
    if split_rest:
        late_over_early = split_rest["late_over_early"]
        drift_channel = max(late_over_early, key=lambda key: abs(float(late_over_early[key]) - 1.0))
        drift_ratio = float(late_over_early[drift_channel])
        if drift_ratio >= 1.5 or drift_ratio <= 0.67:
            lines.append(
                f"The largest split-half drift appears on {drift_channel}: late REST is {drift_ratio:.2f}x early REST, "
                "which suggests contact, impedance, motion, or state drift rather than stable physiology alone."
            )
        else:
            lines.append("Early-vs-late REST drift is modest across channels.")

    action_stats = summary["action_stats"]
    open_delta = np.array([action_stats["OPEN"]["delta_vs_rest_log"][c] for c in channels], dtype=np.float32)
    close_delta = np.array([action_stats["CLOSE"]["delta_vs_rest_log"][c] for c in channels], dtype=np.float32)
    similarity = _cosine_similarity(open_delta, close_delta)
    if similarity >= 0.9:
        lines.append(
            f"OPEN and CLOSE have highly similar rest-relative alpha patterns (cosine similarity {similarity:.2f}), "
            "so alpha topography alone is unlikely to separate those actions cleanly."
        )
    else:
        lines.append(
            f"OPEN and CLOSE have partially distinct rest-relative alpha patterns (cosine similarity {similarity:.2f})."
        )

    dominant_open_drop_channel = channels[int(np.argmin(open_delta))]
    dominant_open_rise_channel = channels[int(np.argmax(open_delta))]
    dominant_close_drop_channel = channels[int(np.argmin(close_delta))]
    dominant_close_rise_channel = channels[int(np.argmax(close_delta))]
    lines.append(
        f"Relative to REST, OPEN is strongest as decreased {dominant_open_drop_channel} power with increased {dominant_open_rise_channel}; "
        f"CLOSE shows the same dominant drop on {dominant_close_drop_channel} and rise on {dominant_close_rise_channel}."
    )

    finger_stats = summary["finger_stats"]
    finger_names = [name for name in FINGER_NAMES.values() if name in finger_stats and name != "NONE"]
    finger_raw = np.array(
        [[finger_stats[name]["raw_mean"][c] for c in channels] for name in finger_names],
        dtype=np.float32,
    )
    channel_ranges = finger_raw.max(axis=0) - finger_raw.min(axis=0)
    range_order = np.argsort(channel_ranges)[::-1]
    top_range_channel = channels[int(range_order[0])]
    next_range_channel = channels[int(range_order[1])] if len(channels) > 1 else channels[int(range_order[0])]
    lines.append(
        f"Finger-level variation is largest on {top_range_channel} and then {next_range_channel}; "
        "AF7/AF8 change much less, so finger discrimination is being driven mostly by lateral channels, not a broad scalp pattern."
    )

    if dominant_ratio >= 2.0 and split_rest and float(split_rest["late_over_early"].get(dominant_channel, 1.0)) >= 1.5:
        lines.append(
            "Interpret the absolute maps as qualitative only. The more defensible views are the rest-relative delta and z-score panels, "
            "which discount baseline asymmetry and make class differences easier to inspect."
        )
    else:
        lines.append("Absolute and rest-relative maps tell a reasonably consistent story, with less evidence of severe baseline distortion.")

    return lines


def write_summary_report(
    *,
    out_path: Path,
    summary_json_path: Path,
    bandpower: np.ndarray,
    y_action: np.ndarray,
    y_finger: np.ndarray,
    channel_names: np.ndarray,
    band: tuple[float, float],
    generated_files: list[str],
) -> None:
    summary = _pairwise_group_stats(bandpower, y_action, y_finger, channel_names)
    interpretations = _interpret_results(summary, channel_names)
    payload = {
        "band_hz": {"low": float(band[0]), "high": float(band[1])},
        "channels": channel_names.tolist(),
        "summary": summary,
        "interpretations": interpretations,
        "generated_files": generated_files,
    }
    summary_json_path.write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# Experimental Muse Topomap Summary",
        "",
        f"Band: {band[0]:.1f}-{band[1]:.1f} Hz",
        "",
        "## Key Findings",
        "",
    ]
    for line in interpretations:
        md_lines.append(f"- {line}")
    md_lines.extend(
        [
            "",
            "## REST Baseline",
            "",
            f"- Raw mean: `{json.dumps(summary['rest_raw_mean'])}`",
            f"- Raw std: `{json.dumps(summary['rest_raw_std'])}`",
            "",
            "## Action Summary",
            "",
        ]
    )
    for action_name in ACTION_NAMES.values():
        if action_name not in summary["action_stats"]:
            continue
        stats = summary["action_stats"][action_name]
        md_lines.append(f"### {action_name}")
        md_lines.append(f"- N: {stats['n']}")
        md_lines.append(f"- Raw mean: `{json.dumps(stats['raw_mean'])}`")
        md_lines.append(f"- Delta vs REST (log10 power): `{json.dumps(stats['delta_vs_rest_log'])}`")
        md_lines.append(f"- Z vs REST: `{json.dumps(stats['z_vs_rest'])}`")
        split = summary["split_halves"].get(action_name)
        if split is not None:
            md_lines.append(f"- Early raw mean: `{json.dumps(split['early_raw_mean'])}`")
            md_lines.append(f"- Late raw mean: `{json.dumps(split['late_raw_mean'])}`")
            md_lines.append(f"- Late / early: `{json.dumps(split['late_over_early'])}`")
        md_lines.append("")

    md_lines.extend(["## Finger Summary", ""])
    for finger_name in FINGER_NAMES.values():
        if finger_name not in summary["finger_stats"]:
            continue
        stats = summary["finger_stats"][finger_name]
        md_lines.append(f"### {finger_name}")
        md_lines.append(f"- N: {stats['n']}")
        md_lines.append(f"- Raw mean: `{json.dumps(stats['raw_mean'])}`")
        md_lines.append(f"- Delta vs REST (log10 power): `{json.dumps(stats['delta_vs_rest_log'])}`")
        md_lines.append(f"- Z vs REST: `{json.dumps(stats['z_vs_rest'])}`")
        md_lines.append("")

    md_lines.extend(["## Generated Files", ""])
    for filename in generated_files:
        md_lines.append(f"- `{filename}`")
    out_path.write_text("\n".join(md_lines))


def render_figure(
    *,
    out_path: Path,
    bandpower: np.ndarray,
    y_action: np.ndarray,
    y_finger: np.ndarray,
    channel_names: np.ndarray,
    band: tuple[float, float],
    spec: FigureSpec,
    blur_sigma: float,
    robust_quantile: float,
) -> Path:
    values_windows = transform_bandpower(bandpower, y_action, spec.metric)
    maps, ncols = build_maps(
        values_windows,
        y_action=np.asarray(y_action, dtype=np.int64),
        y_finger=np.asarray(y_finger, dtype=np.int64),
        group_by=spec.group_by,
        include_none=bool(spec.include_none),
        split_halves=bool(spec.split_halves),
    )
    if not maps:
        raise ValueError(f"No panels available for {spec.filename}")

    metric_cfg = METRIC_CONFIG[spec.metric]
    vmin, vmax = compute_map_limits(
        maps,
        robust_quantile=float(robust_quantile),
        center_zero=bool(metric_cfg["center_zero"]),
    )
    fig = plot_muse_topomap_grid(
        maps,
        channel_names,
        ncols=ncols,
        suptitle=make_figure_title(
            group_by=spec.group_by,
            metric=spec.metric,
            band=band,
            split_halves=spec.split_halves,
            include_none=spec.include_none,
        ),
        cmap=str(metric_cfg["cmap"]),
        blur_sigma=float(blur_sigma),
        vmin=float(vmin),
        vmax=float(vmax),
        colorbar_label=str(metric_cfg["colorbar_label"]),
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Experimental Muse 2 scalp topomaps from eeg_windows.npz. "
            "This remains separate from the main figure pipeline."
        )
    )
    parser.add_argument("--npz", required=True, help="Path to eeg_windows.npz")
    parser.add_argument("--out", default=None, help="Output image path for single-figure mode.")
    parser.add_argument("--out-dir", default=None, help="Output directory for --suite mode.")
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Generate the default improved topomap suite plus a markdown/json interpretation report.",
    )
    parser.add_argument(
        "--group-by",
        choices=("action", "finger", "joint"),
        default="action",
        help="How to aggregate windows into topomap panels for single-figure mode.",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(METRIC_CONFIG.keys()),
        default="log_absolute",
        help="Metric to plot in single-figure mode.",
    )
    parser.add_argument(
        "--split-halves",
        action="store_true",
        help="For action groups, render a 2x3 early/late panel layout.",
    )
    parser.add_argument(
        "--include-none",
        action="store_true",
        help="Include finger NONE when --group-by finger is used.",
    )
    parser.add_argument("--band-low", type=float, default=8.0, help="Lower band edge in Hz.")
    parser.add_argument("--band-high", type=float, default=12.0, help="Upper band edge in Hz.")
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=0.0,
        help="Post-interpolation Gaussian blur sigma. Default is 0.0 to avoid oversmoothing four-channel Muse maps.",
    )
    parser.add_argument(
        "--robust-quantile",
        type=float,
        default=0.05,
        help="Quantile clipping used for figure color limits.",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional markdown report path. When --suite is used and omitted, a default report is written.",
    )
    parser.add_argument(
        "--summary-json-out",
        default=None,
        help="Optional JSON summary path. When --suite is used and omitted, a default summary is written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    npz_path = Path(args.npz).expanduser().resolve()
    X, y_action, y_finger, meta = load_sequence_npz(npz_path)
    channel_names = _channel_names_from_meta(meta)
    fs = _scalar_float(meta.get("target_fs", meta.get("fs")), 256.0)
    band = (float(args.band_low), float(args.band_high))
    bandpower = compute_bandpower_windows(
        X,
        fs,
        band=band,
        channel_count=channel_names.size,
    )

    if args.suite:
        if not args.out_dir:
            raise SystemExit("--suite requires --out-dir")
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []
        for spec in default_suite_specs(band):
            out_path = out_dir / spec.filename
            render_figure(
                out_path=out_path,
                bandpower=bandpower,
                y_action=np.asarray(y_action, dtype=np.int64),
                y_finger=np.asarray(y_finger, dtype=np.int64),
                channel_names=channel_names,
                band=band,
                spec=spec,
                blur_sigma=float(args.blur_sigma),
                robust_quantile=float(args.robust_quantile),
            )
            print(f"Wrote {out_path}")
            generated_files.append(out_path.name)

        band_slug = _band_slug(band)
        summary_md = (
            Path(args.summary_out).expanduser().resolve()
            if args.summary_out
            else out_dir / f"experimental_muse_{band_slug}_summary.md"
        )
        summary_json = (
            Path(args.summary_json_out).expanduser().resolve()
            if args.summary_json_out
            else out_dir / f"experimental_muse_{band_slug}_summary.json"
        )
        write_summary_report(
            out_path=summary_md,
            summary_json_path=summary_json,
            bandpower=bandpower,
            y_action=np.asarray(y_action, dtype=np.int64),
            y_finger=np.asarray(y_finger, dtype=np.int64),
            channel_names=channel_names,
            band=band,
            generated_files=generated_files,
        )
        print(f"Wrote {summary_md}")
        print(f"Wrote {summary_json}")
        return 0

    if not args.out:
        raise SystemExit("single-figure mode requires --out")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec = FigureSpec(
        filename=out_path.name,
        group_by=str(args.group_by),
        metric=str(args.metric),
        include_none=bool(args.include_none),
        split_halves=bool(args.split_halves),
    )
    render_figure(
        out_path=out_path,
        bandpower=bandpower,
        y_action=np.asarray(y_action, dtype=np.int64),
        y_finger=np.asarray(y_finger, dtype=np.int64),
        channel_names=channel_names,
        band=band,
        spec=spec,
        blur_sigma=float(args.blur_sigma),
        robust_quantile=float(args.robust_quantile),
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
