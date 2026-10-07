"""Regenerate the shipped per-format B=1000 TsT-RF removal list from VSI-Bench (downloads data; ~3 min)."""

from pathlib import Path

import pytest

SHIPPED = Path(__file__).resolve().parents[2] / "reproduce" / "data" / "ibp" / "rf" / "pf_b1000"


@pytest.mark.network
@pytest.mark.slow
def test_per_format_b1000_matches_shipped_list(tmp_path):
    from TsT.debiasing.__main__ import main

    out = tmp_path / "pf_b1000"
    assert (
        main(["-b", "vsi", "--alloc", "per_format", "--budget", "1000", "--revision", "bc96b17", "-o", str(out)]) == 0
    )
    got = sorted((out / "removed_ids.txt").read_text().split())
    assert got == sorted((SHIPPED / "removed_ids.txt").read_text().split())
