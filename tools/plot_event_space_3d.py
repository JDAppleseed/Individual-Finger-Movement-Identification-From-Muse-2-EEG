#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.event_space_3d import (
    SUPPORTED_COLOR_MODES,
    SUPPORTED_EMBEDDING_SOURCES,
    SUPPORTED_REDUCERS,
    SUPPORTED_SAMPLE_STRATEGIES,
    build_plot_figure,
    export_frame_to_npz,
    prepare_event_space_dataframe,
    resolve_event_space_artifacts,
)


def _normalize_filter_values(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                out.append(item)
    return out or None


def _resolve_output_path(path_text: str, *, session_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (session_dir / path).resolve()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot EEG windows in 3D embedding space with model predictions, "
            "deployment metadata, and optional event trajectories."
        )
    )
    parser.add_argument("--run-dir", type=str, default=None, help="Model run directory or winning_model snapshot.")
    parser.add_argument("--model-path", type=str, default=None, help="Explicit model checkpoint path.")
    parser.add_argument("--config-path", type=str, default=None, help="Explicit train_config.json path.")
    parser.add_argument("--dataset-npz", type=str, default=None, help="Processed eeg_windows.npz path.")
    parser.add_argument(
        "--infer-config-path",
        type=str,
        default=None,
        help="Optional infer.json path for deployment-style postprocess defaults.",
    )
    parser.add_argument(
        "--replay-manifest-path",
        type=str,
        default=None,
        help="Optional pseudo-live replay_manifest.json path for exact runtime defaults.",
    )
    parser.add_argument(
        "--embedding-source",
        type=str,
        default="latent",
        choices=SUPPORTED_EMBEDDING_SOURCES,
        help="Representation source before 3D reduction.",
    )
    parser.add_argument(
        "--reducer",
        type=str,
        default="pca",
        choices=SUPPORTED_REDUCERS,
        help="3D reducer. UMAP requires umap-learn.",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="true_finger",
        choices=SUPPORTED_COLOR_MODES,
        help="Point coloring mode.",
    )
    parser.add_argument(
        "--connect-trajectories",
        action="store_true",
        help="Connect windows from the same event/trial in temporal order.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Maximum number of plotted points after filtering.",
    )
    parser.add_argument(
        "--sample-strategy",
        type=str,
        default="stratified_joint",
        choices=SUPPORTED_SAMPLE_STRATEGIES,
        help="Sampling strategy used when --max-points truncates the filtered set.",
    )
    parser.add_argument("--seed", type=int, default=43, help="Sampling and reducer random seed.")
    parser.add_argument(
        "--split-filter",
        type=str,
        default="all",
        choices=("all", "train", "test", "unknown"),
        help="Restrict the plot to a dataset split when cached split labels are available.",
    )
    parser.add_argument(
        "--subject-filter",
        action="append",
        default=None,
        help="Repeatable subject filter. Comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--session-filter",
        action="append",
        default=None,
        help="Repeatable session filter. Comma-separated values are also accepted.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device for inference.")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size.")
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=25,
        help="UMAP n_neighbors when --reducer umap is used.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist when --reducer umap is used.",
    )
    parser.add_argument("--output-html", type=str, default=None, help="Write the interactive figure to HTML.")
    parser.add_argument("--output-csv", type=str, default=None, help="Write plotted rows and metadata to CSV.")
    parser.add_argument("--output-npz", type=str, default=None, help="Write plotted rows and metadata to NPZ.")
    parser.add_argument("--title", type=str, default=None, help="Optional custom plot title.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive figure in a browser.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    subject_filters = _normalize_filter_values(args.subject_filter)
    session_filters = _normalize_filter_values(args.session_filter)

    artifacts = resolve_event_space_artifacts(
        run_dir=args.run_dir,
        model_path=args.model_path,
        config_path=args.config_path,
        dataset_npz=args.dataset_npz,
        infer_config_path=args.infer_config_path,
        replay_manifest_path=args.replay_manifest_path,
    )
    frame, summary, notes = prepare_event_space_dataframe(
        artifacts=artifacts,
        embedding_source=args.embedding_source,
        reducer=args.reducer,
        max_points=args.max_points,
        sample_strategy=args.sample_strategy,
        seed=args.seed,
        split_filter=args.split_filter,
        subject_filters=subject_filters,
        session_filters=session_filters,
        device_name=args.device,
        batch_size=args.batch_size,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    if (
        args.connect_trajectories
        and summary["counts"]["display_rows"] < summary["counts"]["filtered_rows"]
    ):
        notes.append(
            "Trajectory lines connect sampled windows only; sampling can skip intermediate points."
        )

    title = args.title or (
        "EEG Event Space 3D "
        f"| source={args.embedding_source} reducer={args.reducer} "
        f"color={args.color_by} n={summary['counts']['display_rows']}"
    )
    fig = build_plot_figure(
        frame,
        color_by=args.color_by,
        connect_trajectories=bool(args.connect_trajectories),
        title=title,
    )

    if args.output_html:
        output_html = _resolve_output_path(
            args.output_html,
            session_dir=artifacts.session_dir,
        )
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html))
        print(f"Saved HTML: {output_html}")
    if args.output_csv:
        output_csv = _resolve_output_path(
            args.output_csv,
            session_dir=artifacts.session_dir,
        )
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
        print(f"Saved CSV: {output_csv}")
    if args.output_npz:
        output_npz = _resolve_output_path(
            args.output_npz,
            session_dir=artifacts.session_dir,
        )
        export_frame_to_npz(
            output_npz,
            frame,
            embedding_source=args.embedding_source,
            reducer=args.reducer,
        )
        print(f"Saved NPZ: {output_npz}")

    print(
        "Prepared 3D event space "
        f"from {summary['counts']['dataset_rows']} dataset rows, "
        f"{summary['counts']['filtered_rows']} filtered rows, "
        f"{summary['counts']['display_rows']} displayed rows."
    )
    print(
        "Artifacts: "
        f"dataset={summary['artifact_paths']['dataset_npz']} "
        f"session={summary['artifact_paths']['session_dir']} "
        f"run={summary['artifact_paths']['run_dir']}"
    )
    for note in notes:
        print(f"- {note}")

    if not args.no_show:
        fig.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
