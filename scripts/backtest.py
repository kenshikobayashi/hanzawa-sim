"""2020年版との照合（バックテスト、PLAN.md 7章）。

2020年版の物語は2013年版の最終回から1〜2年後の出来事である。シミュレーションの3半期後（2015年4月）と
4半期後（2015年10月）の状態で、劇中の事実が成り立つ確率を求める。2020年版の内容はモデルの入力に使って
いないため、これは事後の照合になる（ただし LLM 自体が2020年版を知っている可能性は残る。7章の漏れの検査を参照）。

  uv run python scripts/backtest.py --policy sonnet --runs 20000
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from hanzawa_sim.agents.policy import load_policy_set
from hanzawa_sim.sim.engine import Run
from hanzawa_sim.world.loader import load_scenario
from hanzawa_sim.world.model import Org, Track
from hanzawa_sim.world.params import Params

ROOT = Path(__file__).resolve().parents[1]
SEED = 20131001
IN = (Track.HQ, Track.BRANCH)

# (説明, 判定関数)。判定は名前付きの人物の状態 dict（rank, track, org）と頭取の pid を受け取る
CANON = [
    ("半沢が出向から銀行本体に復帰（2020年版: 営業第二部 次長）", lambda s, td: s["hanzawa"][1] in IN),
    ("半沢が銀行本体の次長", lambda s, td: s["hanzawa"][1] in IN and s["hanzawa"][0] == 3),
    ("渡真利が次長（融資部企画グループ次長）", lambda s, td: s["tomari"][1] in IN and s["tomari"][0] == 3),
    ("近藤が次長（広報部 次長）", lambda s, td: s["kondo"][1] in IN and s["kondo"][0] == 3),
    ("大和田が取締役のまま", lambda s, td: s["owada"][1] in IN and s["owada"][0] == 6),
    ("中野渡が頭取のまま", lambda s, td: td == "nakanowatari"),
    ("内藤が部長のまま（営業第二部長）", lambda s, td: s["naito"][1] in IN and s["naito"][0] == 4),
    ("福山が次長のまま", lambda s, td: s["fukuyama"][1] in IN and s["fukuyama"][0] == 3),
    ("黒崎が金融庁に在籍", lambda s, td: s["kurosaki"][2] == Org.FSA and s["kurosaki"][1] == Track.EXTERNAL),
]


def _worker(args):
    seed, ids, policy, steps = args
    sc = load_scenario()
    pids = [p["id"] for p in sc.persons]
    pol = load_policy_set(policy, pids) if policy else None
    p = Params()
    out = {t: np.zeros(len(CANON)) for t in steps}
    for rid in ids:
        run = (
            Run(sc, p, pol, seed, rid)
            if pol
            else Run(sc, p, __import__("hanzawa_sim.agents.policy").agents.policy.PolicySet(), seed, rid)
        )
        for t in range(max(steps) + 1):
            run.step(t)
            if t in steps:
                s = {q.pid: (q.rank, q.track, q.org) for q in run.w.persons if q.named}
                td = run.w.toudori()
                out[t] += [float(f(s, td.pid if td else "")) for _, f in CANON]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="sonnet")
    ap.add_argument("--runs", type=int, default=20000)
    args = ap.parse_args()
    policy = None if args.policy == "heuristic" else ROOT / "data" / "policies" / f"{args.policy}.json"
    steps = (
        2,
        3,
    )  # 2015年4月時点（3半期後）、2015年10月時点（4半期後）。step t の後の時点は 2013.75 + 0.5(t+1)
    n_proc = os.cpu_count() or 1
    chunks = [list(range(i, args.runs, n_proc * 4)) for i in range(n_proc * 4)]
    tot = {t: np.zeros(len(CANON)) for t in steps}
    with ProcessPoolExecutor(n_proc) as ex:
        for o in ex.map(_worker, [(SEED, c, policy, steps) for c in chunks]):
            for t in steps:
                tot[t] += o[t]
    res = {}
    for t in steps:
        ym = Params().start_time + 0.5 * (t + 1)
        res[f"{int(ym)}年{10 if ym % 1 >= 0.5 else 4}月"] = {
            name: round(100 * v / args.runs, 1) for (name, _), v in zip(CANON, tot[t], strict=True)
        }
    out = ROOT / "data" / "results" / args.policy / "backtest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(k)
        for name, x in v.items():
            print(f"  {x:5.1f}%  {name}")


if __name__ == "__main__":
    main()
