from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from apps.cli.train_daily_baseline import build_parser as build_train_parser
from src.daily_corpus.domain import FEATURE_VERSION, REACTION_VERSION
from src.events.domain.v3 import rules_v3_fingerprint
from src.holdout_evaluation.domain import EXPECTED_RULES_FINGERPRINT
from src.predictive_baseline.application import dataset_readiness
from src.predictive_baseline.data import (
    deterministic_purged_temporal_split,
    load_daily_predictive_dataset,
)
from src.predictive_baseline.domain import (
    DEVELOPMENT_WARNING,
    NON_PERFORMANCE_WARNING,
    NON_TRADING_WARNING,
    BaselineConfig,
    LoadedDataset,
    PredictiveRow,
    RunMode,
    TrainingGate,
    assert_feature_names_safe,
    training_gate,
)
from src.predictive_baseline.modeling import (
    DeterministicPreprocessor,
    classification_metrics,
    regression_metrics,
    train_predictive_baselines,
)
from src.predictive_baseline.reporting import write_training_artifacts
from src.shared.config.settings import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_THINK


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (99, TrainingGate.TRAINING_BLOCKED),
        (100, TrainingGate.PILOT_TRAINING_ALLOWED),
        (500, TrainingGate.BASELINE_EXPERIMENT_ALLOWED),
        (1000, TrainingGate.BASELINE_TRAINING_READY),
    ],
)
def test_training_gate_thresholds(rows: int, expected: TrainingGate) -> None:
    assert training_gate(rows) == expected


def test_real_training_is_blocked_below_100_without_error() -> None:
    result = train_predictive_baselines(_dataset(34), BaselineConfig(), mode=RunMode.REAL)
    assert result.status == "TRAINING_BLOCKED"
    assert result.classification_status == "NOT_TRAINED"
    assert result.regression_status == "NOT_TRAINED"
    assert result.model_binary is None
    assert result.split is None


def test_development_smoke_must_be_explicit() -> None:
    parser = build_train_parser()
    assert parser.parse_args([]).development_smoke is False
    assert parser.parse_args(["--development-smoke"]).development_smoke is True


def test_smoke_output_is_clearly_non_trading_and_not_persisted() -> None:
    result = _smoke_result()
    assert result.status == DEVELOPMENT_WARNING
    assert result.warnings == (
        DEVELOPMENT_WARNING,
        NON_TRADING_WARNING,
        NON_PERFORMANCE_WARNING,
    )
    assert result.model_binary is None


def test_random_split_is_not_configurable() -> None:
    assert "shuffle" not in BaselineConfig.__dataclass_fields__
    source = (Path(__file__).parents[2] / "src" / "predictive_baseline" / "data.py").read_text(
        encoding="utf-8"
    )
    assert "train_test_split" not in source
    assert "random.shuffle" not in source


def test_temporal_split_is_deterministic_for_input_order() -> None:
    rows = _rows(36)
    first = deterministic_purged_temporal_split(rows, BaselineConfig())
    second = deterministic_purged_temporal_split(tuple(reversed(rows)), BaselineConfig())
    assert first.split_sha256 == second.split_sha256
    assert first.assignments() == second.assignments()


def test_train_is_older_than_validation_and_validation_older_than_test() -> None:
    split = deterministic_purged_temporal_split(_rows(36), BaselineConfig())
    assert max(row.publication_date for row in split.train) < min(
        row.publication_date for row in split.validation
    )
    assert max(row.publication_date for row in split.validation) < min(
        row.publication_date for row in split.test
    )


def test_equal_publication_dates_do_not_cross_splits() -> None:
    rows = list(_rows(20))
    rows.append(replace(rows[10], news_id=UUID(int=5000)))
    split = deterministic_purged_temporal_split(tuple(rows), BaselineConfig())
    assignments = split.assignments()
    assert assignments[rows[10].news_id] == assignments[UUID(int=5000)]


def test_overlapping_label_window_is_purged() -> None:
    config = replace(BaselineConfig(), embargo_days=0)
    initial = deterministic_purged_temporal_split(_rows(30), config)
    boundary = min(row.publication_date for row in initial.validation)
    crossing = initial.train[-1]
    changed = tuple(
        replace(row, target_session_date=boundary) if row.news_id == crossing.news_id else row
        for row in _rows(30)
    )
    split = deterministic_purged_temporal_split(changed, config)
    assert crossing.news_id in split.purged_news_ids
    assert crossing.news_id not in split.assignments()


def test_embargo_removes_immediate_boundary_rows() -> None:
    rows = _rows(30, spacing_days=2)
    without = deterministic_purged_temporal_split(
        rows,
        replace(BaselineConfig(), embargo_days=0, purge_overlapping_labels=False),
    )
    with_embargo = deterministic_purged_temporal_split(
        rows,
        replace(BaselineConfig(), embargo_days=1, purge_overlapping_labels=False),
    )
    assert len(with_embargo.embargoed_news_ids) > 0
    assert len(with_embargo.validation) + len(with_embargo.test) < len(without.validation) + len(
        without.test
    )


@pytest.mark.parametrize(
    "name",
    ["abnormal_return", "future_volume", "target_close", "price_target"],
)
def test_future_and_target_feature_columns_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=r"target|future"):
        assert_feature_names_safe({name: 1.0})


def test_targets_are_physically_excluded_from_x() -> None:
    row = _row(0)
    preprocessor = DeterministicPreprocessor.create(("signal", "missing"))
    preprocessor.fit((row,))
    assert "abnormal_return" not in preprocessor.feature_names()
    assert "target_session_date" not in preprocessor.feature_names()


def test_encoder_and_imputer_fit_train_only() -> None:
    train = (_row(0, ticker="ROSN", signal=1.0), _row(1, ticker="ROSN", signal=3.0))
    validation = (_row(2, ticker="NEVER_SEEN", signal=999.0),)
    preprocessor = DeterministicPreprocessor.create(("signal", "missing")).fit(train)
    preprocessor.transform(validation)
    assert preprocessor.fitted_news_ids == tuple(str(row.news_id) for row in train)
    assert preprocessor.numeric_medians["signal"] == 2.0
    assert "NEVER_SEEN" not in preprocessor.categories["ticker"]


def test_unseen_category_maps_to_stable_unknown_bucket() -> None:
    train = (_row(0, ticker="ROSN"), _row(1, ticker="YDEX"))
    unseen = (_row(2, ticker="SBER"),)
    preprocessor = DeterministicPreprocessor.create(("signal", "missing")).fit(train)
    first = preprocessor.transform(unseen)
    second = preprocessor.transform(unseen)
    assert first == second
    unknown_index = preprocessor.feature_names().index("ticker=__UNKNOWN__")
    assert first[0][unknown_index] == 1.0


def test_same_seed_produces_same_predictions() -> None:
    first = _smoke_result()
    second = _smoke_result()
    assert [item.payload() for item in first.predictions] == [
        item.payload() for item in second.predictions
    ]


def test_classification_pipeline_and_naive_baseline_metrics_exist() -> None:
    result = _smoke_result()
    test_metrics = result.classification_metrics["test"]
    assert set(test_metrics) == {"model", "naive_majority"}
    assert "accuracy" in test_metrics["model"]
    assert "balanced_accuracy" in test_metrics["model"]
    assert "macro_f1" in test_metrics["model"]
    assert "log_loss" in test_metrics["model"]
    assert "brier_score" in test_metrics["model"]


def test_regression_pipeline_and_naive_baseline_metrics_exist() -> None:
    result = _smoke_result()
    test_metrics = result.regression_metrics["test"]
    assert {"model", "naive_train_mean", "naive_zero"} <= set(test_metrics)
    assert {"mae", "rmse", "r2", "pearson", "spearman"} <= set(test_metrics["model"])


def test_probability_output_contract_is_complete_and_not_a_recommendation() -> None:
    prediction = _smoke_result().predictions[0].payload()
    required = {
        "news_id",
        "ticker",
        "prediction_time",
        "model_version",
        "target_horizon",
        "predicted_direction",
        "prob_up",
        "prob_flat",
        "prob_down",
        "predicted_abnormal_return",
        "model_probability",
        "dataset_version",
    }
    assert set(prediction) == required
    assert (
        abs(
            float(prediction["prob_up"])
            + float(prediction["prob_flat"])
            + float(prediction["prob_down"])
            - 1.0
        )
        < 1e-12
    )
    assert "recommendation" not in prediction


def test_metric_helpers_cover_confusion_and_regression() -> None:
    classification = classification_metrics(
        ["DOWN", "FLAT", "UP"],
        ["DOWN", "UP", "UP"],
        [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.1, 0.1, 0.8]],
    )
    regression = regression_metrics([0.0, 1.0, 2.0], [0.0, 1.5, 1.5])
    assert classification["confusion_matrix"]["FLAT"]["UP"] == 1
    assert regression["mae"] > 0


def test_artifact_manifest_contains_fingerprints_and_smoke_has_no_model(tmp_path: Path) -> None:
    dataset = _dataset(36)
    result = train_predictive_baselines(dataset, BaselineConfig(), mode=RunMode.DEVELOPMENT_SMOKE)
    paths = write_training_artifacts(
        tmp_path,
        dataset=dataset,
        config=BaselineConfig(),
        result=result,
        git_sha="abc123",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert result.split is not None
    assert manifest["dataset_sha256"] == dataset.dataset_sha256
    assert manifest["feature_schema_sha256"] == dataset.feature_schema_sha256
    assert manifest["split_sha256"] == result.split.split_sha256
    assert "model" not in paths


def test_artifact_run_cannot_silently_overwrite(tmp_path: Path) -> None:
    dataset = _dataset(36)
    result = _smoke_result()
    config = BaselineConfig()
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    write_training_artifacts(
        tmp_path,
        dataset=dataset,
        config=config,
        result=result,
        git_sha="abc123",
        created_at=created_at,
    )
    with pytest.raises(FileExistsError):
        write_training_artifacts(
            tmp_path,
            dataset=dataset,
            config=config,
            result=result,
            git_sha="abc123",
            created_at=created_at,
        )


def test_test_is_not_used_for_tuning_and_calibration_is_blocked() -> None:
    result = _smoke_result()
    assert result.test_used_for_tuning is False
    assert result.calibration_status == "NOT_READY_INSUFFICIENT_VALIDATION"


def test_readiness_service_reports_current_gate_and_paths() -> None:
    payload = dataset_readiness(_dataset(34), intraday_feature_ready=21)
    assert payload["daily_feature_ready"] == 34
    assert payload["intraday_feature_ready"] == 21
    assert payload["rows_to_100"] == 66
    assert payload["rows_to_500"] == 466
    assert payload["rows_to_1000"] == 966
    assert payload["training_gate"] == "TRAINING_BLOCKED"
    assert payload["date_range"] == {"from": "2025-01-01", "to": "2025-04-10"}


def test_loader_keeps_target_outside_numeric_features(tmp_path: Path) -> None:
    features = tmp_path / "features.jsonl"
    reactions = tmp_path / "reactions.jsonl"
    _write_loader_fixture(features, reactions)
    dataset = load_daily_predictive_dataset(features, reactions)
    assert dataset.rows[0].abnormal_return == 0.01
    assert "abnormal_return" not in dataset.rows[0].numeric_features
    assert dataset.rows[0].target_session_date == date(2026, 1, 3)


def test_feature_availability_cannot_be_after_baseline_session() -> None:
    row = _row(1)
    late = datetime.combine(row.publication_date, datetime.min.time(), UTC)
    with pytest.raises(ValueError, match="after the baseline"):
        replace(row, prediction_time=late).validate()


def test_real_gate_can_train_only_after_threshold_and_produces_versioned_binary() -> None:
    result = train_predictive_baselines(_dataset(120), BaselineConfig(), mode=RunMode.REAL)
    assert result.status == "SUCCEEDED"
    assert result.model_binary is not None
    assert result.model_binary_sha256 == hashlib.sha256(result.model_binary).hexdigest()


def test_zero_cost_pipeline_has_no_network_or_hosted_training_dependency() -> None:
    root = Path(__file__).parents[2] / "src" / "predictive_baseline"
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(root.glob("*.py")))
    lowered = source.lower()
    assert "httpx" not in lowered
    assert "requests." not in lowered
    assert "openai" not in lowered
    assert '"paid_ml_api_used": true' not in lowered


def test_frozen_nlp_and_daily_reaction_identities_are_unchanged() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert DEFAULT_OLLAMA_MODEL == "qwen3.5:9b"
    assert DEFAULT_OLLAMA_THINK is False
    assert FEATURE_VERSION == "ml-daily-features-v1"
    assert REACTION_VERSION == "date-safe-daily-reaction-v1"


def _dataset(count: int) -> LoadedDataset:
    rows = _rows(count)
    return LoadedDataset(
        rows=rows,
        dataset_sha256="d" * 64,
        feature_schema_sha256="f" * 64,
        numeric_feature_names=("missing", "signal"),
        dataset_version="synthetic-unit-test",
    )


def _rows(count: int, *, spacing_days: int = 3) -> tuple[PredictiveRow, ...]:
    return tuple(_row(index, spacing_days=spacing_days) for index in range(count))


def _row(
    index: int,
    *,
    ticker: str | None = None,
    signal: float | None = None,
    spacing_days: int = 3,
) -> PredictiveRow:
    publication = date(2025, 1, 1) + timedelta(days=index * spacing_days)
    targets = (-0.01, 0.0, 0.01)
    return PredictiveRow(
        news_id=UUID(int=index + 1),
        ticker=ticker or ("ROSN" if index % 2 == 0 else "YDEX"),
        source="ISSUER_RSS_A" if index % 2 == 0 else "ISSUER_RSS_B",
        timestamp_quality="EXACT",
        event_type=("DIVIDEND", "OTHER", "FINANCIAL_RESULTS")[index % 3],
        publication_date=publication,
        baseline_session_date=publication - timedelta(days=1),
        target_session_date=publication + timedelta(days=1),
        prediction_time=datetime.combine(publication - timedelta(days=1), datetime.min.time(), UTC)
        + timedelta(hours=18),
        numeric_features={
            "signal": float(index) if signal is None else signal,
            "missing": None if index % 4 == 0 else float(index % 5),
        },
        abnormal_return=targets[index % 3],
        dataset_version="synthetic-unit-test",
    )


def _smoke_result():
    return train_predictive_baselines(
        _dataset(36), BaselineConfig(), mode=RunMode.DEVELOPMENT_SMOKE
    )


def _write_loader_fixture(feature_path: Path, reaction_path: Path) -> None:
    feature = {
        "metadata": {
            "news_id": str(UUID(int=1)),
            "ticker": "ROSN",
            "source": "ROSNEFT_PRESS_RELEASES_RSS",
            "timestamp_quality": "EXACT",
            "publication_date": "2026-01-02",
            "baseline_session_date": "2025-12-30",
            "feature_available_at": "2025-12-30T20:49:59+00:00",
            "feature_version": FEATURE_VERSION,
            "reaction_version": REACTION_VERSION,
        },
        "features": {"baseline_security_close": "100", "baseline_imoex_close": "1000"},
        "labels": {"abnormal_return": "0.01"},
    }
    reaction = {
        "news_id": str(UUID(int=1)),
        "ticker": "ROSN",
        "source": "ROSNEFT_PRESS_RELEASES_RSS",
        "timestamp_quality": "EXACT",
        "publication_date": "2026-01-02",
        "target_session_date": "2026-01-03",
        "abnormal_return": "0.01",
        "reaction_version": REACTION_VERSION,
    }
    feature_path.write_text(json.dumps(feature) + "\n", encoding="utf-8")
    reaction_path.write_text(json.dumps(reaction) + "\n", encoding="utf-8")
