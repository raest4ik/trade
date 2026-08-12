from __future__ import annotations

import inspect

from src.corpus_quality.application import load_publication_time_records


def test_cumulative_quality_loader_supports_bounded_growth_to_1000() -> None:
    source = inspect.getsource(load_publication_time_records)
    assert "1 <= limit <= 1000" in source
    assert ".limit(limit)" in source
