import numpy as np

from utils.sequence_data import summarize_windows


def test_summarize_windows_uses_low_redundancy_feature_set():
    X = np.array(
        [
            [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]],
            [[4.0, 1.0], [3.0, 1.0], [2.0, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )

    df = summarize_windows(X)

    assert list(df.columns) == [
        "ch1_mean",
        "ch1_std",
        "ch1_line_length",
        "ch1_zero_cross_rate",
        "ch2_mean",
        "ch2_std",
        "ch2_line_length",
        "ch2_zero_cross_rate",
    ]
    assert df.shape == (2, 8)
    assert np.isclose(df.loc[0, "ch1_mean"], 2.5)
    assert np.isclose(df.loc[0, "ch1_line_length"], 1.0)
    assert np.isclose(df.loc[0, "ch2_zero_cross_rate"], 1.0 / 3.0)
