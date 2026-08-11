# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression, Ridge


class LogisticRegressionAdapter:
    def __init__(self, *, random_seed: int) -> None:
        self._model: Any = LogisticRegression(
            random_state=random_seed,
            max_iter=2000,
            solver="lbfgs",
        )

    def fit(self, features: list[list[float]], targets: list[str]) -> None:
        self._model.fit(features, targets)

    def predict(self, features: list[list[float]]) -> list[str]:
        return [str(value) for value in self._model.predict(features)]

    def predict_probabilities(self, features: list[list[float]]) -> list[list[float]]:
        return [
            [float(probability) for probability in row]
            for row in self._model.predict_proba(features)
        ]

    def classes(self) -> list[str]:
        return [str(value) for value in self._model.classes_]


class RidgeRegressionAdapter:
    def __init__(self, *, alpha: float) -> None:
        self._model: Any = Ridge(alpha=alpha)

    def fit(self, features: list[list[float]], targets: list[float]) -> None:
        self._model.fit(features, targets)

    def predict(self, features: list[list[float]]) -> list[float]:
        return [float(value) for value in self._model.predict(features)]
