from bayes_opt import BayesianOptimization
import numpy as np

class PrunableBayesianOptimizer:
    def __init__(self, f, pbounds, random_state=42, patience=30):
        self.optimizer = BayesianOptimization(
            f=f,
            pbounds=pbounds,
            random_state=random_state,
            allow_duplicate_points=True
        )
        self._max_no_improve = patience
        self._history = []
        self._best_score = -np.inf
        self._no_improve_cnt = 0

    def maximize(self, init_points=10, n_iter=100):
        def _callback(res):
            score = res["target"]
            self._history.append(score)

            if score > self._best_score:
                self._best_score = score
                self._no_improve_cnt = 0
            else:
                self._no_improve_cnt += 1

            if self._no_improve_cnt >= self._max_no_improve:
                print(f"[EarlyStopping] No improvement in {self._max_no_improve} rounds.")
                raise KeyboardInterrupt()

        try:
            self.optimizer.maximize(init_points=init_points, n_iter=n_iter, callback=_callback)
        except KeyboardInterrupt:
            pass

        self.max = self.optimizer.max
