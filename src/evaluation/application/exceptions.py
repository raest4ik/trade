from __future__ import annotations


class EvaluationError(Exception):
    pass


class EvaluationDatasetNotFoundError(EvaluationError):
    pass


class AnnotationDatasetValidationError(EvaluationError):
    pass


class EvaluationThresholdError(EvaluationError):
    pass
