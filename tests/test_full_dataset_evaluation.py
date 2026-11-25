from pathlib import Path
import numpy as np
import pytest
from tools.evaluate_full_dataset import evaluate


def test_full_dataset_diagnostic_requires_subject_selection_for_mixed_data(tmp_path):
    p = tmp_path / 'data.npz'
    np.savez(p, X=np.zeros((2, 64, 4)), y_action=[0, 0], y_finger=[0, 0], subject_id=['A', 'B'])
    with pytest.raises(ValueError, match='Multiple subjects'):
        evaluate(p, tmp_path)


def test_full_dataset_diagnostic_rejects_empty_subject(tmp_path):
    p = tmp_path / 'data.npz'
    np.savez(p, X=np.zeros((2, 64, 4)), y_action=[0, 0], y_finger=[0, 0], subject_id=['A', 'A'])
    with pytest.raises(ValueError, match='No windows match'):
        evaluate(p, tmp_path, 'B')


def test_full_dataset_diagnostic_rejects_all_nonfinite(tmp_path):
    p = tmp_path / 'data.npz'
    np.savez(p, X=np.full((2, 64, 4), np.nan), y_action=[0, 0], y_finger=[0, 0], subject_id=['A', 'A'])
    with pytest.raises(ValueError, match='No finite windows'):
        evaluate(p, tmp_path)
