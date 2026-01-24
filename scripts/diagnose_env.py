#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import sys
import time
import traceback
from importlib import metadata


def _print_section(title: str) -> None:
    print(f"{title}:")


def _print_kv(label: str, value: str) -> None:
    print(f"  {label}: {value}")


def _print_torch_metadata() -> None:
    _print_section("Torch package metadata")
    try:
        dist = metadata.distribution("torch")
    except metadata.PackageNotFoundError:
        _print_kv("status", "not installed")
        return
    except Exception as exc:
        _print_kv("status", f"error reading metadata: {exc}")
        return

    name = dist.metadata.get("Name", "torch")
    summary = dist.metadata.get("Summary", "")
    location = str(dist.locate_file(""))
    requires = dist.requires or []

    _print_kv("name", name)
    _print_kv("version", dist.version)
    if summary:
        _print_kv("summary", summary)
    _print_kv("location", location)
    if requires:
        _print_kv("requires", ", ".join(sorted(set(requires))))


def _time_torch_import() -> None:
    _print_section("Torch import timing")
    start = time.perf_counter()
    try:
        import torch
    except Exception:
        elapsed = time.perf_counter() - start
        _print_kv("import", f"failed after {elapsed:.2f}s")
        print("exception:")
        traceback.print_exc()
        print("hints:")
        print("- Use Python 3.11 or 3.12 for this repo.")
        print("- Recreate the venv: ./scripts/setup_venv.sh")
        print("- If you are on Python 3.13, torch import may hang or fail.")
        return

    elapsed = time.perf_counter() - start
    _print_kv("import", f"ok ({elapsed:.2f}s)")
    _print_kv("torch.__version__", getattr(torch, "__version__", "unknown"))


def main() -> int:
    _print_section("Python")
    _print_kv("version", sys.version.replace("\n", " "))
    _print_kv("executable", sys.executable)

    _print_section("Platform")
    _print_kv("platform", platform.platform())
    _print_kv("machine", platform.machine())
    _print_kv("mac_ver", str(platform.mac_ver()))

    _print_section("Venv")
    _print_kv("VIRTUAL_ENV", os.environ.get("VIRTUAL_ENV", "(not set)"))

    _print_torch_metadata()
    _time_torch_import()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
