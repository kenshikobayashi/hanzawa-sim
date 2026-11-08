"""モンテカルロの実行と、10年後の役職分布の表示。

--policy を省くと暫定ルール（traits から計算、動作確認用）で実行する。
例: uv run python scripts/run_mc.py --runs 2000 --seed 1 --policy data/policies/sonnet.json
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

from hanzawa_sim.agents.policy import load_policy_set
from hanzawa_sim.sim.runner import CATEGORIES, distribution, run_many
from hanzawa_sim.world.loader import load_scenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--policy", type=Path, default=None)
    args = ap.parse_args()

    sc = load_scenario()
    pids = [p["id"] for p in sc.persons]
    policies = load_policy_set(args.policy, pids) if args.policy else None
    t0 = time.perf_counter()
    results = run_many(args.runs, args.seed, processes=args.procs, policies=policies)
    dt = time.perf_counter() - t0
    n = len(results)
    print(f"{n:,} 試行 / {dt:.1f} 秒（1試行あたり {1000 * dt / n:.2f} ms・並列実行の実時間ベース）")
    if policies is None:
        print("※ 方策は暫定ルール（LLMなし）。数値は世界モデルの動作確認用で、分析結果ではない")
    else:
        print(f"方策: {args.policy}")
    print()
    names = {p["id"]: p["name"] for p in sc.persons}
    half = 1.96 * math.sqrt(0.25 / n) * 100
    print(f"10年後（2023年9月）の状態の分布 [%]（95%信頼区間の半幅は最大 ±{half:.1f} ポイント）")
    header = f"{'人物':<8}" + "".join(f"{c:>10}" for c in CATEGORIES) + f"{'役員経験':>8}"
    print(header)
    for pid, name in names.items():
        if pid == "kurosaki":
            continue
        d = distribution(results, pid)
        ever_exec = sum(r.summary[pid]["max_rank"] >= 5 for r in results) / n * 100
        row = (
            f"{name:<8}"
            + "".join(f"{100 * d.get(c, 0) / n:>10.1f}" for c in CATEGORIES)
            + f"{ever_exec:>8.1f}"
        )
        print(row)

    d = distribution(results, "kurosaki")
    print("\n黒崎駿一: " + ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in d.most_common()))

    final_td = Counter(r.toudori[-1] for r in results)
    named_td = {k: v for k, v in final_td.items() if k in names}
    npc_share = sum(v for k, v in final_td.items() if k not in names)
    print(
        "\n10年後の頭取: "
        + ", ".join(
            f"{names[k]} {100 * v / n:.1f}%" for k, v in sorted(named_td.items(), key=lambda kv: -kv[1])
        )
        + f", NPC {100 * npc_share / n:.1f}%"
    )
    first_change = [
        next((t for t, pid in enumerate(r.toudori) if pid != "nakanowatari"), None) for r in results
    ]
    stayed = sum(x is None for x in first_change)
    changed = sorted(x for x in first_change if x is not None)
    med = changed[len(changed) // 2] * 0.5 if changed else float("nan")
    print(
        f"中野渡の在任: 10年間在任 {100 * stayed / n:.1f}%、退任までの中央値 {med:.1f} 年（退任した試行のみ）"
    )


if __name__ == "__main__":
    main()
