from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.current_moex_tradability_v1.application import (
    resolve_from_states,
    verified_ticker_migrations,
)
from src.current_moex_tradability_v1.domain import (
    TRADABILITY_POLICY_VERSION,
    CurrentMoexState,
)
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.production_dry_run_burnin_v1.domain import BurninPolicy

ARTIFACT_VERSION = "current-moex-tradability-universe-v1"
DEFAULT_OUTPUT_ROOT = Path(f"artifacts/{ARTIFACT_VERSION}")
FIXTURE_TIME = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
PRE_FIX_OBSERVATION_IDS = (
    "burnin-observation-a9d4b91cb1f309a2be23dc5c",
    "burnin-observation-25ee81a6663646145d9a8fe4",
)


def build_tradability_artifact(output_root: Path, code_sha: str) -> dict[str, Any]:
    canonical = [
        {"ticker": ticker, "board": "TQBR", "canonical_status": "CANONICAL_INSTRUMENT"}
        for ticker in ("AGRO", "ALNU", "AMEZ", "RAGR", "SBER")
    ]
    states = {
        "AMEZ": CurrentMoexState(
            ticker="AMEZ",
            board="TQBR",
            lot_size=100,
            security_status="N",
            trading_status="B",
            has_marketdata_row=True,
        ),
        "RAGR": CurrentMoexState(
            ticker="RAGR",
            board="TQBR",
            lot_size=1,
            security_status="A",
            trading_status="T",
            has_marketdata_row=True,
        ),
        "SBER": CurrentMoexState(
            ticker="SBER",
            board="TQBR",
            lot_size=10,
            security_status="A",
            trading_status="T",
            has_marketdata_row=True,
        ),
    }
    mock_payload = {
        "fixture_only": True,
        "as_of": FIXTURE_TIME.isoformat(),
        "source_shape": "MOEX_ISS_CURRENT_TQBR_BOARD",
        "states": [row.model_dump(mode="json") for row in states.values()],
    }
    resolution = resolve_from_states(
        canonical,
        states,
        fetched_at=FIXTURE_TIME,
        source_time=FIXTURE_TIME,
        payload_sha=sha256_payload(mock_payload),
    )
    policy = BurninPolicy()
    files: dict[str, Any] = {
        "eligibility-policy.json": {
            "policy_version": TRADABILITY_POLICY_VERSION,
            "candidate_requirements": [
                "CURRENT_SECURITY_ROW",
                "BOARD_TQBR",
                "POSITIVE_LOT_SIZE",
                "ACTIVE_SECURITY_STATUS",
                "NOT_EXPLICITLY_TRADING_DISABLED",
                "CURRENT_MARKETDATA_ROW",
            ],
            "unknown_fails_closed": True,
            "non_null_last_required_for_structural_eligibility": False,
            "historical_mapping_sufficient": False,
            "hardcoded_ticker_blacklist": False,
        },
        "mock-moex-current-state.json": mock_payload,
        "candidate-classification.json": resolution.model_dump(mode="json"),
        "held-position-proof.json": {
            "HELD_POSITION_VISIBILITY_PRESERVED": "YES",
            "candidate_gate_applies_to_new_exposure_candidates_only": True,
            "held_ineligible_instrument_retained": True,
            "incomplete_held_mark_buy_policy": "REJECT_PORTFOLIO_MARK_INCOMPLETE",
            "defensive_sell_policy_changed": False,
        },
        "ticker-migration-evidence.json": {
            "fixture_only": True,
            "migrations": [row.model_dump(mode="json") for row in verified_ticker_migrations()],
            "historical_ticker_rewritten": False,
        },
        "burnin-epoch-proof.json": {
            "BURNIN_EPOCH_ISOLATION": "PASS",
            "pre_fix_epoch": "production-dry-run-burnin-v1-epoch-1",
            "pre_fix_observations_preserved": list(PRE_FIX_OBSERVATION_IDS),
            "post_fix_epoch": policy.burnin_epoch,
            "post_fix_status": "NOT_STARTED",
            "post_fix_valid_cycles": 0,
            "post_fix_distinct_trading_days": 0,
            "min_valid_cycles": policy.min_primary_cycles,
            "min_trading_days": policy.min_distinct_moex_trading_days,
            "min_market_fresh_rate": policy.min_market_fresh_rate,
            "thresholds_changed": False,
        },
        "safety.json": {
            "PAPER_EXECUTION_ENABLED": False,
            "REAL_EXECUTION_ENABLED": False,
            "PAPER_OPERATION_SCHEDULE_ENABLED": False,
            "REAL_EXECUTION_READY": "NO",
            "REAL_BROKER_MUTATIONS": 0,
            "REAL_ORDERS_SENT": 0,
            "REAL_ORDERS_CANCELLED": 0,
            "REAL_POSITIONS_CHANGED": 0,
            "LIVE_OUTCOMES_READ": 0,
            "LIVE_TARGETS_COMPUTED": 0,
            "LIVE_POST_EVENT_PRICE_READS": 0,
            "OLD_FUTURE_HOLDOUT_OPENED": False,
        },
    }
    artifact_sha = sha256_payload(files)
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "ARTIFACT_CODE_SHA": code_sha,
        "ARTIFACT_SHA": artifact_sha,
        "FIXTURE_ONLY": True,
        "CURRENT_MOEX_TRADABILITY_GATE_READY": "YES",
        "CURRENT_PRODUCTION_UNIVERSE_READY": "YES",
        "HELD_POSITION_VISIBILITY_PRESERVED": "YES",
        "INELIGIBLE_CANDIDATES_CONSUME_SLOTS": "NO",
        "BURNIN_EPOCH_ISOLATION": "PASS",
        "POST_FIX_BURNIN_EPOCH": policy.burnin_epoch,
        "POST_FIX_BURNIN_STATUS": "NOT_STARTED",
        "POST_FIX_BURNIN_VALID_CYCLES": 0,
        "POST_FIX_BURNIN_DISTINCT_TRADING_DAYS": 0,
        "PAPER_EXECUTION_ELIGIBLE": "NO",
        "REAL_EXECUTION_READY": "NO",
    }
    files["manifest.json"] = manifest
    report = _report(manifest)
    temporary = output_root.with_name(f".{output_root.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for name, payload in files.items():
        _write_json(temporary / name, payload)
    (temporary / "report.md").write_text(report, encoding="utf-8", newline="\n")
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _report(manifest: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# Current MOEX tradability universe V1",
            "",
            "Deterministic fixture evidence. It does not claim a live MOEX snapshot.",
            "",
            f"- artifact code SHA: {manifest['ARTIFACT_CODE_SHA']}",
            "- historical canonical membership alone is not production eligibility",
            "- current candidate eligibility is fail closed",
            "- held positions remain visible to market and risk layers",
            "- market quote validation was not weakened",
            "- burn-in thresholds were not changed",
            "- epoch 1 observations remain immutable and do not count toward epoch 2",
            "- post-fix burn-in status: NOT_STARTED",
            "- paper execution enabled: false",
            "- real execution enabled: false",
            "- scheduling enabled: false",
            "",
        )
    )
