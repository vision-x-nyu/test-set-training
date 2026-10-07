"""CPU reproduction of the IBP tables and the GPT-4o blind baselines from shipped data."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT)


def test_ibp_tables_check():
    if not (ROOT / "reproduce" / "ibp_tables.py").exists():
        pytest.skip("reproduce/ not present")
    out = _run("reproduce/ibp_tables.py", "--check")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "OK: all" in out.stdout


OFFLINE = ("--benchmarks", "vsi", "cvb", "mmmu")  # video_mme downloads its gold answers


@pytest.mark.parametrize("unparseable", ["wrong", "random"])
def test_gpt4o_blind_rescore_check(unparseable):
    out = _run("scripts/blind_baselines.py", *OFFLINE, "--unparseable", unparseable, "--check")
    assert out.returncode == 0, out.stdout + out.stderr


def test_gpt4o_default_matches_the_paper():
    out = _run("scripts/blind_baselines.py", *OFFLINE).stdout
    for bm, printed in [("vsi", "33.98"), ("cvb", "44.84"), ("mmmu", "52.30")]:
        assert any(line.startswith(bm) and f"top-1  {printed}" in line for line in out.splitlines()), out


@pytest.mark.network
def test_gpt4o_video_mme_with_downloaded_gold_answers():
    out = _run("scripts/blind_baselines.py", "--benchmarks", "video_mme", "--check")
    assert out.returncode == 0 and "46.59" in out.stdout, out.stdout + out.stderr


def test_gpt4o_random_option_is_order_independent():
    a = _run("scripts/blind_baselines.py", "--unparseable", "random", "--benchmarks", "vsi", "cvb").stdout.splitlines()
    b = _run("scripts/blind_baselines.py", "--unparseable", "random", "--benchmarks", "cvb", "vsi").stdout.splitlines()
    assert sorted(a) == sorted(b) and len(a) == 2


def test_gpt4o_api_run_needs_a_key(tmp_path):
    import os

    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    out = subprocess.run(
        [sys.executable, "scripts/blind_baselines.py", "--run", "--benchmarks", "cvb", "--output_dir", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert out.returncode == 2 and "OPENAI_API_KEY" in out.stderr
