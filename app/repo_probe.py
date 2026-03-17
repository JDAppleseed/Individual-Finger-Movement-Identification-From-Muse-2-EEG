from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ArgInfo:
    flags: List[str]
    dest: Optional[str]
    default: Any
    help: str
    action: Optional[str] = None
    arg_type: Optional[str] = None
    choices: Optional[List[Any]] = None


@dataclass
class ScriptInfo:
    path: Path
    args: List[ArgInfo]
    constants: Dict[str, Any]


STEP_SCRIPTS = {
    "step1": "1_stream_and_record.py",
    "step1b": "1b_extract_windows.py",
    "event_review": "5_review_events.py",
    "event_validate": "5_validate_events.py",
    "train": "2_train_model.py",
    "topomaps": "tools/experimental_muse_topomaps.py",
    "evaluate": "3_evaluate_model.py",
    "evaluate_deepchecks": "3b_deepchecks_evaluate.py",
    "evaluate_figures": "3c_live_paper_figures.py",
    "evaluate_reports": "4_generate_reports.py",
    "live_infer": "7_live_infer_and_actuate.py",
    "live_review": "tools/analyze_live_predictions.py",
    "diagnostics": "tools/check_time_alignment.py",
}


def _safe_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def parse_constants(path: Path) -> Dict[str, Any]:
    constants: Dict[str, Any] = {}
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return constants

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    value = _safe_literal(node.value)
                    if value is not None:
                        constants[target.id] = value
    return constants


def parse_argparse(path: Path) -> List[ArgInfo]:
    args: List[ArgInfo] = []
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return args

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        flags = []
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                flags.append(arg.value)
        if not flags:
            continue
        dest = None
        default = None
        help_text = ""
        action = None
        arg_type = None
        choices = None
        for kw in node.keywords:
            if kw.arg == "dest":
                dest = _safe_literal(kw.value)
            elif kw.arg == "default":
                default = _safe_literal(kw.value)
            elif kw.arg == "help":
                help_text = _safe_literal(kw.value) or ""
            elif kw.arg == "action":
                action = _safe_literal(kw.value)
            elif kw.arg == "type":
                if isinstance(kw.value, ast.Name):
                    arg_type = kw.value.id
                else:
                    arg_type = _safe_literal(kw.value)
            elif kw.arg == "choices":
                choices = _safe_literal(kw.value)
        if dest is None:
            long_flags = [f for f in flags if f.startswith("--")]
            if long_flags:
                dest = long_flags[0].lstrip("-").replace("-", "_")
        args.append(
            ArgInfo(
                flags=flags,
                dest=dest,
                default=default,
                help=help_text,
                action=action,
                arg_type=arg_type,
                choices=choices,
            )
        )
    return args


def parse_script(path: Path) -> Optional[ScriptInfo]:
    if not path.exists():
        return None
    return ScriptInfo(
        path=path,
        args=parse_argparse(path),
        constants=parse_constants(path),
    )


def discover_scripts(repo_root: Path) -> Dict[str, ScriptInfo]:
    results: Dict[str, ScriptInfo] = {}
    for key, rel in STEP_SCRIPTS.items():
        path = repo_root / rel
        info = parse_script(path)
        if info:
            results[key] = info
    return results
