"""test_benchmark_grouping.py — cross-seed aggregation in optim.benchmark (D-EVID-13).

`build_table` promises "one row per (config x target), averaged over seeds". It keyed the
grouping on `config_id`, and `Config.config_id()` hashes the WHOLE config INCLUDING
`train.seed` — which `optim/sweep.py` and `optim/target_report.py` set per seed. So every
K-seed replicate became K groups of one, `n_seeds` was always 1, and
`kstar_identifiability_std` was always NaN. The suite never caught it because no test ran
more than one seed.

These tests run synthetic index rows through the real `build_table`, so a regression shows
up as a grouping failure rather than as a plausible-looking table.
"""
from __future__ import annotations

import copy
import json
import math

import pytest

from rngrn.config import Config
from rngrn.optim.benchmark import build_table


def _row(*, seed, arm_id="arm_a", sample_key="sample_0000", kstar=1.0, turing=True,
         dataset="three_gene_val", **extra):
    """A minimal run-index row carrying the fields build_table reads."""
    r = {
        "run_id": f"run_{arm_id}_{sample_key}_{seed}",
        "seed": seed,
        # config_id is what SHOULD vary per seed — that is the whole point
        "config_id": f"cfg_{arm_id}_{sample_key}_{seed}",
        "arm_id": arm_id,
        "sample_key": sample_key,
        "source": "registry",
        "dataset_label": dataset,
        "N": 3, "m": 3, "form": "competitive", "strategy": "fixed",
        "recovered_turing": turing,
        "kstar_model": kstar,
        "sign_match_frac": 0.8,
        "loss": 0.5,
        "kstar_fft_rel_err": 0.05,
        "trivial_kstar_fft_err": 0.08,
        "kstar_rel_err": 0.03,
        "trivial_kstar_err": 0.001,
    }
    r.update(extra)
    return r


def _table(tmp_path, rows):
    (tmp_path / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return build_table(str(tmp_path), backend="jsonl")


# ======================================================================================
# Config.arm_id — the seed-independent identity the grouping needs
# ======================================================================================
def test_arm_id_is_invariant_to_the_seed():
    a = copy.deepcopy(Config()); a.train.seed = 0
    b = copy.deepcopy(Config()); b.train.seed = 1
    assert a.config_id() != b.config_id(), "precondition: config_id DOES vary with seed"
    assert a.arm_id() == b.arm_id(), "arm_id must not vary with train.seed"


def test_arm_id_separates_a_PINNED_init_arm_from_a_free_init_arm():
    """model.seed=None ("derive from train.seed") and model.seed=7 ("hold the init FIXED
    while train.seed varies") are DIFFERENT EXPERIMENTS, not seed replicates of one.

    Pooling them would mix a degenerate zero cross-seed spread into a real one — which is
    the shape of D-EVID-4, where a constant model.seed=0 made K "replicates" one draw.
    """
    a = copy.deepcopy(Config()); a.model.seed = None
    b = copy.deepcopy(Config()); b.model.seed = 7
    assert a.arm_id() != b.arm_id(), "a pinned-init arm must not pool with a free-init arm"


def test_arm_id_is_constant_across_seeds_within_a_PINNED_init_arm():
    """Within the pinned arm, train.seed still varies and must still group."""
    cfgs = []
    for s in (0, 1, 2):
        c = copy.deepcopy(Config()); c.model.seed = 7; c.train.seed = s
        cfgs.append(c)
    assert len({c.arm_id() for c in cfgs}) == 1
    assert len({c.config_id() for c in cfgs}) == 3, "precondition: config_id still varies"


def test_arm_id_is_constant_across_seeds_within_a_FREE_init_arm():
    cfgs = []
    for s in (0, 1, 2):
        c = copy.deepcopy(Config()); c.model.seed = None; c.train.seed = s
        cfgs.append(c)
    assert len({c.arm_id() for c in cfgs}) == 1


def test_arm_id_still_separates_genuinely_different_configs():
    """It must not become so coarse that different science pools into one row."""
    base = Config()
    for mutate in (
        lambda c: setattr(c.train, "adam_steps", c.train.adam_steps + 100),
        lambda c: c.loss.weights.__setitem__("resid", 0.6),
        lambda c: setattr(c.data, "sample_key", "sample_0009"),
        lambda c: setattr(c.model, "N", 4),
    ):
        other = copy.deepcopy(base)
        mutate(other)
        assert other.arm_id() != base.arm_id(), f"arm_id collided after {mutate}"


# ======================================================================================
# build_table grouping
# ======================================================================================
def test_seed_replicates_group_into_one_row(tmp_path):
    """THE DEFECT: three seeds of one arm must be ONE row with n_seeds == 3."""
    rows = [_row(seed=s, kstar=1.0 + 0.1 * s) for s in (0, 1, 2)]
    table = _table(tmp_path, rows)

    assert len(table) == 1, f"expected one group, got {len(table)}: {table}"
    assert table[0]["n_seeds"] == 3
    # identifiability is the spread ACROSS seeds — it needs > 1 member to exist at all
    assert not math.isnan(table[0]["kstar_identifiability_std"])
    assert table[0]["kstar_identifiability_std"] > 0


def test_the_seeds_themselves_are_recorded(tmp_path):
    """n_seeds alone cannot be audited; the actual seeds must be on the row."""
    table = _table(tmp_path, [_row(seed=s) for s in (2, 0, 1)])
    assert json.loads(table[0]["seeds"]) == [0, 1, 2], "seeds must be recorded, sorted"


def test_different_sample_keys_do_not_pool(tmp_path):
    """Two TARGETS are two rows. Pooling them would be a worse defect than the one fixed.

    `sample_key` never appeared in the old group key at all — it was carried only via
    `config_id`. Dropping config_id without replacing that identity would have merged
    different targets into one mean.
    """
    rows = ([_row(seed=s, sample_key="sample_0000", arm_id="arm_a") for s in (0, 1)]
            + [_row(seed=s, sample_key="sample_0007", arm_id="arm_a") for s in (0, 1)])
    table = _table(tmp_path, rows)
    assert len(table) == 2, f"targets must not pool; got {len(table)} rows"
    assert {r["n_seeds"] for r in table} == {2}
    assert {r["sample_key"] for r in table} == {"sample_0000", "sample_0007"}


def test_different_arms_do_not_pool(tmp_path):
    """Different configs stay separate — the property config_id was protecting."""
    rows = ([_row(seed=s, arm_id="arm_a") for s in (0, 1)]
            + [_row(seed=s, arm_id="arm_b") for s in (0, 1)])
    table = _table(tmp_path, rows)
    assert len(table) == 2
    assert {r["arm_id"] for r in table} == {"arm_a", "arm_b"}


def test_different_datasets_do_not_pool(tmp_path):
    rows = ([_row(seed=s, dataset="three_gene_val") for s in (0, 1)]
            + [_row(seed=s, dataset="two_gene_classical_val") for s in (0, 1)])
    assert len(_table(tmp_path, rows)) == 2


def test_legacy_rows_without_arm_id_do_not_silently_pool(tmp_path):
    """Rows recorded before `arm_id` existed carry no seed-independent identity.

    They fall back to `config_id`, which means they still split per seed — the old
    behaviour. That is the honest degradation: those rows genuinely lack the information
    needed to group them, and inventing a grouping would risk pooling different configs.
    The row says so via `arm_id = None`.
    """
    rows = []
    for s in (0, 1):
        r = _row(seed=s)
        del r["arm_id"]
        rows.append(r)
    table = _table(tmp_path, rows)
    assert len(table) == 2, "legacy rows must NOT be pooled on a guess"
    assert all(r["arm_id"] is None for r in table)
    assert all(r["n_seeds"] == 1 for r in table)


def test_reproducibility_table_still_runs(tmp_path):
    """`reproducibility_table` has NO callers — which is exactly why it needs a test.

    It shares `_group_key` with `build_table`. Widening that key to an 8-tuple for
    D-EVID-13 left this function unpacking 7, so it raised `ValueError: too many values to
    unpack` on EVERY call — while the commit message claimed it "inherits the fix". Nothing
    caught it: no caller, no test, suite green. Dead code must not be allowed to become
    broken code silently; if it is deleted, delete this test with it.
    """
    from rngrn.optim.benchmark import (reproducibility_table, reproducibility_markdown,
                                       REPRODUCIBILITY_COLUMNS)
    rows = [_row(seed=s, kstar=1.0 + 0.1 * s) for s in (0, 1)]
    for r in rows:                       # the function filters on this key
        r["repro_sign_vector"] = json.dumps([1, -1, 0, 0, 1, -1, -1, 0, 1])
        r["repro_N"] = 3
        r["repro_kstar"] = r["kstar_model"]
        r["repro_D_ratio"] = 10.0
    (tmp_path / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    table = reproducibility_table(str(tmp_path), backend="jsonl")
    assert len(table) == 1 and table[0]["K"] == 2
    assert table[0]["arm_id"] == "arm_a"
    # every declared column must actually be produced
    for col in REPRODUCIBILITY_COLUMNS:
        assert col in table[0], f"REPRODUCIBILITY_COLUMNS declares {col}, row lacks it"
    assert isinstance(reproducibility_markdown(table), str)


def test_identifiability_std_is_nan_for_a_single_seed(tmp_path):
    """One seed has no spread. NaN, not 0.0 — a fake 0.0 would read as perfect agreement."""
    table = _table(tmp_path, [_row(seed=0)])
    assert table[0]["n_seeds"] == 1
    assert math.isnan(table[0]["kstar_identifiability_std"])


def test_identifiability_std_uses_only_turing_reaching_seeds(tmp_path):
    """Pre-existing contract, pinned here because the grouping change moves this code."""
    rows = [_row(seed=0, kstar=1.0, turing=True),
            _row(seed=1, kstar=1.2, turing=True),
            _row(seed=2, kstar=99.0, turing=False)]   # must not enter the spread
    table = _table(tmp_path, rows)
    assert table[0]["n_seeds"] == 3
    assert table[0]["kstar_identifiability_std"] == pytest.approx(0.1)
