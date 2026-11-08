import numpy as np
import pytest

from hanzawa_sim.agents.distill import Answer, evaluate, fit_softmax
from hanzawa_sim.agents.features import N_FEATURES
from hanzawa_sim.agents.policy import DistilledPolicy, PolicySet
from hanzawa_sim.agents.special_cards import (
    ANON_MAP,
    DECIDERS,
    HR_FEATURES,
    RESPONSE_FEATURES,
    anonymize,
    hr_cards,
    repro_cards,
    response_cards,
    sanction_cards,
)
from hanzawa_sim.sim.engine import run_one
from hanzawa_sim.world.loader import load_scenario
from hanzawa_sim.world.model import N_ACTIONS
from hanzawa_sim.world.params import Params


@pytest.fixture(scope="module")
def sc():
    return load_scenario()


def test_special_cards(sc):
    pids = [p["id"] for p in sc.persons]
    rc = response_cards(sc, pids, n_each=1, seed=3, n_runs=5)
    cards = [c for cs in rc.values() for c in cs]
    assert cards and all(len(c.x) == len(RESPONSE_FEATURES) for c in cards)
    assert all("選べる応答" in c.text for c in cards)
    sj = sanction_cards(sc, DECIDERS, 8, 3)
    assert all(len(v) == 8 for v in sj.values())
    hc = hr_cards(sc, DECIDERS, 6, 3)
    for cs in hc.values():
        for c in cs:
            assert len(c.x) == len(HR_FEATURES) and sum(v != 0 for v in c.x) >= 2
    assert len(repro_cards()) >= 8


def test_anonymize_removes_names(sc):
    from hanzawa_sim.agents.persona import load_personas, system_prompt

    text = anonymize(system_prompt(sc, "hanzawa", load_personas()))
    for p in sc.persons:
        assert p["name"] not in text
    for k in ("東京中央銀行", "伊勢島ホテル", "大和田", "半沢"):
        assert k in ANON_MAP and k not in text


def test_fit_softmax_recovers_parameters():
    rng = np.random.default_rng(0)
    K, F, n = 4, 3, 400
    W_true = rng.normal(size=(K, F + 1))
    X = rng.normal(size=(n, F))
    Z = np.hstack([np.ones((n, 1)), X]) @ W_true.T
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    M = np.ones((n, K), dtype=bool)
    m = fit_softmax(X, P, M, ["a"] * n, ["f0", "f1", "f2"], list("ABCD"), 1e-6, 1e-6)
    W = m.W("a")
    # ソフトマックスは行の定数ずらしで不変なので、平均を引いて比べる
    assert np.allclose(W - W.mean(0), W_true - W_true.mean(0), atol=0.05)
    data = [Answer("", "", "action", "a", "", "named", False, "val", X[i], M[i], P[i], "") for i in range(20)]
    assert evaluate(m, data) < 0.01


def test_engine_runs_with_distilled_policies(sc):
    rng = np.random.default_rng(1)
    pids = [p["id"] for p in sc.persons]
    named = {
        pid: DistilledPolicy(rng.normal(size=(N_ACTIONS, 1 + N_FEATURES)), rng.normal(size=(3, 1 + 7)))
        for pid in pids
    }
    sanction = {d: rng.normal(size=(4, 6)) for d in DECIDERS}
    hr = {d: rng.normal(size=len(HR_FEATURES)) for d in DECIDERS}
    ps = PolicySet(named, sanction, hr, name="random")
    n_distilled = 0
    for rid in range(30):
        res = run_one(sc, Params(), 9, rid, ps, keep_events=True, debug=True)
        n_distilled += sum(1 for e in res.events if e.etype == "SANCTION" and e.detail.get("distilled"))
    assert n_distilled > 0


def test_llm_run_with_fake_backend(sc):
    from hanzawa_sim.agents.elicit import ACTION_PROB_KEYS
    from hanzawa_sim.agents.llm import LLMResult
    from hanzawa_sim.agents.llm_run import LLMRun

    class Fake:
        calls = 0

        def call(self, tag, system_prompt, user_prompt, schema):
            Fake.calls += 1
            assert "これまでの出来事" in user_prompt
            probs = dict.fromkeys(ACTION_PROB_KEYS, 0)
            probs.update({"work": 60, "gather": 20, "support": 20})
            return LLMResult(True, {"answers": [{"card_id": "x", "probs": probs, "reason": "r"}]}, {}, 0.0, 0)

    rng = np.random.default_rng(2)
    pids = [p["id"] for p in sc.persons]
    named = {
        pid: DistilledPolicy(rng.normal(size=(N_ACTIONS, 1 + N_FEATURES)), rng.normal(size=(3, 1 + 7)))
        for pid in pids
    }
    ps = PolicySet(named, name="random")
    run = LLMRun(sc, Params(), ps, 9, 0, backend=Fake(), prompts=dict.fromkeys(pids, "sp"), distilled=ps)
    for t in range(4):
        run.step(t)
    assert Fake.calls == len(run.records) > 0
    assert all(r["source"] == "llm" and r["action"] in ("WORK", "GATHER", "SUPPORT") for r in run.records)
