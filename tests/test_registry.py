"""test_registry.py — dataset store + pluggable index backend.

Covers: register a synthetic HDF5 payload -> list it -> load a sample THROUGH the
firewall gate (observable reaches RecoveryInput, truth is quarantined in AnswerKey).
Runs on BOTH index backends (jsonl, sqlite) to prove they are interchangeable.
"""
import numpy as np
import h5py
import pytest


def _make_payload(path, n=3, N=3, H=16):
    """A tiny HDF5 payload: per-sample final_frame + quarantined jacobian/x_star."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for i in range(n):
            g = f.create_group(f"sample_{i:04d}")
            g.create_dataset("final_frame", data=rng.standard_normal((N, H, H)).astype("float32"))
            g.create_dataset("jacobian", data=rng.standard_normal((N, N)))
            g.create_dataset("x_star", data=rng.standard_normal(N))
            g.create_dataset("D", data=np.array([1.0, 40.0, 20.0])[:N])
            g.attrs["split"] = "train" if i < 2 else "val"


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_register_list_load(tmp_path, backend):
    from rngrn.data import registry as reg
    from rngrn.data import gate

    payload = tmp_path / "raw.h5"
    _make_payload(str(payload))
    droot = str(tmp_path / "datasets")

    man = reg.register(droot, "toy_v1", str(payload),
                       provenance={"source": "unit-test"}, backend=backend)
    assert man["n_samples"] == 3
    assert man["frame_shape"] == [3, 16, 16]

    listed = reg.list_datasets(droot, backend=backend)
    assert any(r["dataset_id"] == "toy_v1" for r in listed)

    # load a sample through the FIREWALL gate
    ri, ak = gate.from_registry(droot, "toy_v1", "sample_0000", N=3,
                                observed_idx=[0, 1, 2], L=100.0, backend=backend)
    # observable reached recovery...
    assert ri.frame.shape == (3, 16, 16)
    assert ri.N == 3 and ri.observed_idx == (0, 1, 2)
    # ...truth is quarantined in the answer key, NOT in the recovery input
    assert ak.J.shape == (3, 3) and ak.x_star.shape == (3,)
    assert not hasattr(ri, "J") and not hasattr(ri, "x_star")


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_index_roundtrip(tmp_path, backend):
    from rngrn.index import open_index
    idx = open_index(str(tmp_path), "runs", backend)
    idx.append(dict(run_id="r1", kstar_rel_err=0.5, recovered_turing=True))
    idx.append(dict(run_id="r2", kstar_rel_err=0.1, recovered_turing=False))
    rows = idx.read()
    assert len(rows) == 2
    assert {r["run_id"] for r in rows} == {"r1", "r2"}
    # backend-agnostic predicate filter
    good = idx.query(where=lambda r: r.get("recovered_turing"))
    assert [r["run_id"] for r in good] == ["r1"]


def test_registry_not_imported_by_recovery_side():
    """registry.py is ANSWER-KEY side: no recovery-side module may import it."""
    import ast, pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "rngrn"
    recovery_side = ["recover.py", "losses/terms.py", "losses/total.py",
                     "model.py", "observables.py", "eval/rollout.py", "eval/analysis.py"]
    # match the DATASET registry (rngrn.data.registry) specifically — NOT the generic
    # model registry (rngrn.registry), which recovery-side code legitimately uses.
    for rel in recovery_side:
        tree = ast.parse((src / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [a.name for a in node.names]
                mod = node.module or ""
                bad = ("data.registry" in mod) or (mod.endswith("data") and "registry" in names)
                assert not bad, f"{rel} imports data.registry (answer-key side)"
            elif isinstance(node, ast.Import):
                for a in node.names:
                    assert "data.registry" not in a.name, \
                        f"{rel} imports data.registry (answer-key side)"
