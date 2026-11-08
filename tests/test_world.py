import numpy as np
import pytest

from hanzawa_sim.agents.policy import PolicySet, sample
from hanzawa_sim.sim.engine import Intervention, Run, run_one
from hanzawa_sim.sim.rng import P, make_table
from hanzawa_sim.world.loader import build_world, load_scenario
from hanzawa_sim.world.model import EXEC_CAPACITY, Org, Track
from hanzawa_sim.world.params import Params


@pytest.fixture(scope="module")
def sc():
    return load_scenario()


def test_initial_state(sc):
    p = Params()
    w = build_world(sc, p, make_table(0, 0, p.n_steps)[0, P.INIT])
    named = [q for q in w.persons if q.named]
    assert len(named) == 15
    assert w.toudori().pid == "nakanowatari"
    assert w.person("hanzawa").track == Track.SECONDED
    assert w.person("owada").rank == 6
    assert w.person("kurosaki").org == Org.FSA
    for r, cap in EXEC_CAPACITY.items():
        expected = cap - (1 if r == 7 else 0)  # 大和田の降格で空いた常務の席
        assert w.occupancy(r) == expected, r
    assert w.person("owada").scandal == pytest.approx(0.4)  # S1 で処分済み


def test_invariants_hold_over_many_runs(sc):
    p = Params()
    for run_id in range(40):
        run_one(sc, p, seed=7, run_id=run_id, debug=True)


def test_reproducible(sc):
    p = Params()
    a = run_one(sc, p, seed=3, run_id=5, keep_events=True)
    b = run_one(sc, p, seed=3, run_id=5, keep_events=True)
    assert a.toudori == b.toudori
    assert a.summary == b.summary
    assert [(e.step, e.etype, e.actor, e.target, e.outcome) for e in a.events] == [
        (e.step, e.etype, e.actor, e.target, e.outcome) for e in b.events
    ]


def test_intervention_keeps_prefix_identical(sc):
    """共通乱数法: 介入した半期より前の経過は、介入なしの試行と一致する。"""
    p = Params()
    base = run_one(sc, p, seed=11, run_id=2, keep_events=True)
    iv = run_one(sc, p, seed=11, run_id=2, keep_events=True, interventions=[Intervention(6, "expose", "S2")])
    key = lambda e: (e.step, e.etype, e.actor, e.target, e.outcome)
    assert [key(e) for e in base.events if e.step < 6] == [key(e) for e in iv.events if e.step < 6]


def test_expose_intervention_triggers_sanction(sc):
    p = Params()
    run = Run(
        sc,
        p,
        PolicySet(),
        1,
        0,
        [Intervention(0, "expose", "S2")],
    )
    run.step(0)
    asano = run.w.person("asano").idx
    assert any(e.etype == "SANCTION" and e.target == asano for e in run.events)
    assert next(s for s in run.w.secrets if s.sid == "S2").public


def test_sample_respects_mask():
    logits = np.zeros(5)
    mask = np.array([False, True, False, True, False])
    for u in np.linspace(0.001, 1.0, 50):
        assert mask[sample(logits, mask, float(u))]
