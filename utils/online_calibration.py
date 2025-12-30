# utils/online_calibration.py

import numpy as np


class OnlineCalibrator:
    """
    Online confidence & threshold calibration using EMA
    """

    def __init__(
        self,
        init_threshold=0.75,
        min_threshold=0.55,
        max_threshold=0.90,
        ema_alpha=0.05
    ):
        self.threshold = init_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.alpha = ema_alpha

        self.correct_buffer = []

    def update(self, confidence: float, correct: bool):
        """
        Update calibration based on recent prediction outcome
        """

        self.correct_buffer.append(int(correct))
        if len(self.correct_buffer) > 100:
            self.correct_buffer.pop(0)

        acc = np.mean(self.correct_buffer)

        # EMA threshold adaptation
        target = confidence if acc > 0.8 else self.threshold + 0.02
        self.threshold = (
            (1 - self.alpha) * self.threshold + self.alpha * target
        )

        self.threshold = float(
            np.clip(self.threshold, self.min_threshold, self.max_threshold)
        )

    def allow_actuation(self, confidence, uncertainty):
        """
        Safety gate
        """
        return (
            confidence >= self.threshold
            and uncertainty < 0.15
        )
