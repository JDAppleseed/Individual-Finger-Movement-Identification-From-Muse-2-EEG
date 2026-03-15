import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_step3b_module():
    path = Path(__file__).resolve().parents[1] / "3b_deepchecks_evaluate.py"
    spec = importlib.util.spec_from_file_location("step3b_deepchecks_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_deepchecks_split_uses_disjoint_row_ids():
    step3b = _load_step3b_module()
    df = pd.DataFrame({"f1": np.arange(6), "f2": np.arange(10, 16)})
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    lookup_ids = np.arange(6, dtype=np.int64)

    train_df, train_labels, train_lookup = step3b._prepare_deepchecks_split(
        df.iloc[:4], labels[:4], lookup_ids[:4], seed=1, index_start=0
    )
    test_df, test_labels, test_lookup = step3b._prepare_deepchecks_split(
        df.iloc[4:], labels[4:], lookup_ids[4:], seed=2, index_start=len(train_df)
    )

    assert train_df.index.name == "row_id"
    assert test_df.index.name == "row_id"
    assert set(train_df.index).isdisjoint(set(test_df.index))
    assert sorted(train_df.index.tolist()) == list(range(0, len(train_df)))
    assert sorted(test_df.index.tolist()) == list(
        range(len(train_df), len(train_df) + len(test_df))
    )
    assert sorted(train_labels.tolist()) == sorted(labels[:4].tolist())
    assert sorted(test_labels.tolist()) == sorted(labels[4:].tolist())
    assert sorted(train_lookup.tolist()) == sorted(lookup_ids[:4].tolist())
    assert sorted(test_lookup.tolist()) == sorted(lookup_ids[4:].tolist())
