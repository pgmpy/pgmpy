import numpy as np
import pandas as pd

class IGCI:
    def __init__(self):
        pass

    def fit(self, data, ref_measure='gaussian'):
        self.data = data
        self.ref_measure = ref_measure
        
    def estimate_direction(self, x, y):
        x_data = self.data[x].values
        y_data = self.data[y].values

        if self.ref_measure == 'gaussian':
            score_xy = self._gaussian_score(x_data, y_data)
            score_yx = self._gaussian_score(y_data, x_data)
        else:
            score_xy = self._uniform_score(x_data, y_data)
            score_yx = self._uniform_score(y_data, x_data)

        if score_xy < score_yx:
            return f"{x}->{y}", abs(score_yx - score_xy)
        else:
            return f"{y}->{x}", abs(score_xy - score_yx)

    def _uniform_score(self, x, y):
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]

        dx = np.diff(x_sorted)
        dy = np.diff(y_sorted)

        eps = 1e-8
        valid = (np.abs(dx) > eps) & (np.abs(dy) > eps)
        dx = dx[valid]
        dy = dy[valid]

        log_derivatives = np.log(np.abs(dy / dx))
        return np.mean(log_derivatives)

    def _gaussian_score(self, x, y):
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)

        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]

        dx = np.diff(x_sorted)
        dy = np.diff(y_sorted)

        eps = 1e-8
        valid = (np.abs(dx) > eps) & (np.abs(dy) > eps)
        dx = dx[valid]
        dy = dy[valid]

        score = np.sum(dy * np.sign(dx))
        return score / (len(dy) + eps)
