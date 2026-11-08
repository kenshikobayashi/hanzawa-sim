"""人物ごとの状態の推移（毎年9月末時点）を集計する。図「状態の推移」用。

基準シナリオ（analyze.py）と同じシード・試行番号で実行するため、最初の N 回は基準シナリオと同じ試行になる。
  uv run python scripts/trajectories.py --policy sonnet --runs 20000
出力: data/results/<policy>/trajectories_<runs>.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from hanzawa_sim.agents.policy import PolicySet, load_policy_set
from hanzawa_sim.sim.engine import Run
from hanzawa_sim.sim.runner import final_category
from hanzawa_sim.world.loader import load_scenario
from hanzawa_sim.world.params import Params

ROOT = Path(__file__).resolve().parents[1]
SEED = 20131001  # analyze.py と同じ
PIDS = ["hanzawa", "tomari", "kondo", "owada", "fukuyama"]


def _worker(args):
    ids, policy = args
    sc = load_scenario()
    pids = [p["id"] for p in sc.persons]
    pol = load_policy_set(policy, pids) if policy else PolicySet()
    p = Params()
    counts: Counter = Counter()
    for rid in ids:
        run = Run(sc, p, pol, SEED, rid)
        for t in range(p.n_steps):
            run.step(t)
            if t % 2 == 1:  # 9月末
                year = 2014 + t // 2
                for pid in PIDS:
                    q = run.w.person(pid)
                    s = {"rank": q.rank, "track": int(q.track), "org": int(q.org)}
                    counts[(pid, year, final_category(s))] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="sonnet")
    ap.add_argument("--runs", type=int, default=20000)
    args = ap.parse_args()
    policy = None if args.policy == "heuristic" else ROOT / "data" / "policies" / f"{args.policy}.json"
    n_proc = os.cpu_count() or 1
    chunks = [list(range(i, args.runs, n_proc * 4)) for i in range(n_proc * 4)]
    total: Counter = Counter()
    with ProcessPoolExecutor(n_proc) as ex:
        for c in ex.map(_worker, [(ch, policy) for ch in chunks]):
            total.update(c)
    out: dict = {"n": args.runs, "persons": {}}
    for (pid, year, cat), v in sorted(total.items()):
        out["persons"].setdefault(pid, {}).setdefault(str(year), {})[cat] = v
    path = ROOT / "data" / "results" / args.policy / f"trajectories_{args.runs}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
