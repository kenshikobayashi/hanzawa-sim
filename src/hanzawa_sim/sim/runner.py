"""多数回の試行を並列に実行し、10年後の役職を集計する。"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from hanzawa_sim.agents.policy import PolicySet
from hanzawa_sim.sim.engine import Intervention, RunResult, run_one
from hanzawa_sim.world.loader import load_scenario
from hanzawa_sim.world.model import FSA_RANK_TITLES, IN_BANK, TRACK_JA, Org, Track
from hanzawa_sim.world.params import Params

CATEGORIES = (
    "頭取",
    "副頭取・専務・常務",
    "取締役・執行役員",
    "部長",
    "次長以下",
    "出向",
    "転籍",
    "退職・退任",
    "失脚",
)


def final_category(s: dict[str, Any]) -> str:
    track, rank = Track(s["track"]), s["rank"]
    if s["org"] == Org.FSA:
        return f"金融庁:{FSA_RANK_TITLES[rank]}" if track == Track.EXTERNAL else TRACK_JA[track]
    if track in IN_BANK:
        if rank == 10:
            return "頭取"
        if rank >= 7:
            return "副頭取・専務・常務"
        if rank >= 5:
            return "取締役・執行役員"
        return "部長" if rank == 4 else "次長以下"
    return TRACK_JA[track]


def _worker(args: tuple) -> list[RunResult]:
    seed, run_ids, params, policies, interventions, keep_events = args
    sc = load_scenario()
    return [run_one(sc, params, seed, rid, policies, interventions, keep_events) for rid in run_ids]


def run_many(
    n_runs: int,
    seed: int,
    params: Params | None = None,
    processes: int | None = None,
    policies: PolicySet | None = None,
    interventions: Sequence[Intervention] = (),
    keep_events: bool = False,
    start: int = 0,
) -> list[RunResult]:
    params = params or Params()
    processes = processes or os.cpu_count() or 1
    chunk = max(1, min(500, n_runs // (processes * 4) or 1))
    batches = [
        (
            seed,
            list(range(s, min(s + chunk, start + n_runs))),
            params,
            policies,
            tuple(interventions),
            keep_events,
        )
        for s in range(start, start + n_runs, chunk)
    ]
    if processes == 1:
        return [r for b in batches for r in _worker(b)]
    with ProcessPoolExecutor(max_workers=processes) as ex:
        return [r for rs in ex.map(_worker, batches) for r in rs]


def distribution(results: list[RunResult], pid: str) -> Counter:
    return Counter(final_category(r.summary[pid]) for r in results)
