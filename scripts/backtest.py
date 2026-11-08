"""2020年版との照合（バックテスト、PLAN.md 7章）。

2020年版は、日本語版 Wikipedia の基礎情報では時代設定が2020年である。一方、作中の半沢は2013年版の最終回で
命じられた出向先（東京セントラル証券）にまだおり、最終回からの経過期間は作中で明示されていない。そこで次の
2つの時点で、2020年版で描かれた事実が成り立つ確率を求める。
  - 2015年4月（3半期後）: 最終回から1〜2年後の物語と解釈した場合
  - 2020年4月（13半期後）: 時代設定（2020年）どおりと解釈した場合
2020年版の内容はモデルの入力に使っていないため、事後の照合になる（ただし LLM 自体が2020年版を知っている
可能性は残る。7章の漏れの検査を参照）。

  uv run python scripts/backtest.py --policy sonnet --runs 8000
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
    (
        "黒崎が金融行政に在籍（証券取引等監視委員会・金融庁）",
        lambda s, td: s["kurosaki"][2] == Org.FSA and s["kurosaki"][1] == Track.EXTERNAL,
    ),
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
    ap.add_argument("--runs", type=int, default=8000)
    args = ap.parse_args()
    policy = None if args.policy == "heuristic" else ROOT / "data" / "policies" / f"{args.policy}.json"
    steps = (
        2,
        12,
    )  # 2015年4月時点（3半期後）と2020年4月時点（13半期後）。step t の後の時点は 2013.75 + 0.5(t+1)
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
