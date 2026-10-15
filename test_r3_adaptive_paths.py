#!/usr/bin/env python3
"""Small branch and budget regressions for the offline measured analyzer."""
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import r3_cpa_matrix_rank as rank


ROOT = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "r3_onekey_m_rules" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_budget():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "budget.json"
        path.write_text(json.dumps({"rank_word_calls": 0, "rank_word_limit": 1, "exhausted": False}))
        old = os.environ.get("R3_RANK_BUDGET_PATH")
        os.environ["R3_RANK_BUDGET_PATH"] = str(path)
        try:
            rank.charge_rank()
            try:
                rank.charge_rank()
            except RuntimeError as error:
                assert "budget_exceeded" in str(error)
            else:
                raise AssertionError("rank budget did not stop")
            assert json.loads(path.read_text())["exhausted"] is True
        finally:
            if old is None:
                del os.environ["R3_RANK_BUDGET_PATH"]
            else:
                os.environ["R3_RANK_BUDGET_PATH"] = old


def check_ninth_word_backtrack():
    m00 = load("test_m00", "t07-m00-groups.py")
    calls = [0]

    def fake_rank(*args, **kwargs):
        calls[0] += 1
        values = range(16) if calls[0] == 1 else (0,)
        return {"scored_final_beam": [{"value": value, "score": 1.0}
                                        for value in values]}

    m00.PRODUCT.rank = SimpleNamespace(rank_word=fake_rank)
    m00.PRODUCT.limb = object()
    product = np.zeros((4, 6), dtype=np.uint64)
    samples = np.zeros((4, 6))
    narrow, _, _, _ = m00.recover_candidates(product, samples, 8, 16)
    calls[0] = 0
    wide, _, frontier, _ = m00.recover_candidates(product, samples, 16, 32)
    assert 9 not in {item["value"] for item in narrow}
    assert 9 in {item["value"] for item in wide}
    assert frontier[0]["expanded"] == 16
    m00.PRODUCT.rank = SimpleNamespace(rank_word=lambda *_args, **_kwargs:
        {"scored_final_beam": [{"value": 0, "score": 1.0}]})
    cache = {}
    first, first_calls, _, first_hit = m00.recover_candidates(product, samples, 8, 16, cache)
    second, second_calls, _, second_hit = m00.recover_candidates(product, samples, 8, 16, cache)
    assert first == second and first_calls == 6 and second_calls == 1
    assert first_hit is False and second_hit is True


def check_carry_position_paths():
    m10 = load("test_m10", "t08-m10-groups.py")

    def fake_rank(base, *_args, **_kwargs):
        values = (0, 1) if np.all(base == np.uint64(0xffffffff)) else (0,)
        rows = [{"value": value, "score": 1.0} for value in values]
        return {"scored_final_beam": rows, "stages": [{}, {}, {}, {"top_k": rows}]}

    def grid(labels, wave):
        result = np.zeros((labels.shape[1], wave.shape[1]))
        result[0, 5 if labels[0, 0] == 0 else 10] = 1
        return result

    m10.PRODUCT = SimpleNamespace(
        rank=SimpleNamespace(rank_word=fake_rank),
        limb=SimpleNamespace(low=SimpleNamespace(hw32=lambda value: value & np.uint64(255))),
        CONTROL=SimpleNamespace(grid=grid), residual=lambda values, _rows: values)
    product = np.zeros((4, 6), dtype=np.uint64)
    product[:, 0] = np.uint64(0xffffffff)
    traces = np.zeros((4, 500), dtype=np.float32)
    candidates, _, _, _, frontier, _ = m10.rank_sequence_candidates(
        product, None, traces, [{} for _ in range(4)], np.arange(4), 0,
        track_paths=True, word_beam=2, branch_beam=2)
    assert {item["points"][1] for item in candidates} == {9, 14}
    assert len(frontier[1]["location_proposals"]) == 2
    cache = {}
    first, _, _, first_calls, _, first_hit = m10.rank_sequence_candidates(
        product, None, traces, [{} for _ in range(4)], np.arange(4), 0,
        track_paths=True, word_beam=2, branch_beam=2, prefix_cache=cache)
    second, _, _, second_calls, _, second_hit = m10.rank_sequence_candidates(
        product, None, traces, [{} for _ in range(4)], np.arange(4), 0,
        track_paths=True, word_beam=2, branch_beam=2, prefix_cache=cache)
    assert first == second and first_calls > second_calls == 2
    assert first_hit is False and second_hit is True


if __name__ == "__main__":
    check_budget()
    check_ninth_word_backtrack()
    check_carry_position_paths()
    print("adaptive path checks passed")
