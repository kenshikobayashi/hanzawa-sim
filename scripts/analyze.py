"""Phase 5: 基準シナリオのモンテカルロ、介入実験、感度分析。

  uv run python scripts/analyze.py baseline --policy sonnet --runs 100000
  uv run python scripts/analyze.py interventions --policy sonnet --runs 10000
  uv run python scripts/analyze.py sensitivity --policy sonnet --runs 5000

結果は data/results/<policy>/ に JSON で保存する。介入実験と感度分析は、基準シナリオと同じ乱数（共通乱数）で
同じ試行番号を実行し、試行ごとの差（対の差）で効果を推定する。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from hanzawa_sim.agents.policy import load_policy_set
from hanzawa_sim.analysis.collect import Agg, collect
from hanzawa_sim.sim.engine import Intervention
from hanzawa_sim.world.loader import load_scenario
from hanzawa_sim.world.params import Params

ROOT = Path(__file__).resolve().parents[1]
SEED = 20131001

INTERVENTIONS = {
    "S2公表_浅野の5000万円": [Intervention(0, "expose", "S2")],
    "S3公表_黒崎の婚約": [Intervention(0, "expose", "S3")],
    "S4公表_大和田と近藤の取引": [Intervention(0, "expose", "S4")],
    "S5公表_半沢の疎開資料": [Intervention(0, "expose", "S5")],
    "中野渡が直後に退任": [Intervention(0, "retire", "nakanowatari")],
    "大和田が直後に退職": [Intervention(0, "retire", "owada")],
    "半沢が告発しない": [Intervention(0, "forbid", "hanzawa", "EXPOSE")],
    "渡真利が半沢を支援しない": [Intervention(0, "forbid", "tomari", "SUPPORT")],
}

# 感度分析: 不確かなパラメータを基準値の 0.5倍・1.5倍（加算型は ±）に変える
SENSITIVITY = {
    "inspection_rate": ("mul", 0.5, 1.5),
    "leak_rate": ("mul", 0.5, 1.5),
    "case_fail_rate": ("mul", 0.5, 1.5),
    "latent_scandal_rate": ("mul", 0.5, 1.5),
    "board_weight": ("mul", 0.5, 1.5),
    "return_base": ("add", -0.5, 0.5),
    "age_second_rate": ("mul", 0.5, 1.5),
    "toudori_hazard": ("mul", 0.5, 1.5),
    "hr_tau": ("mul", 0.5, 1.5),
}


def policies_for(name: str):
    if name == "heuristic":
        return None
    pids = [p["id"] for p in load_scenario().persons]
    return load_policy_set(ROOT / "data" / "policies" / f"{name}.json", pids)


def ci(k: int, n: int) -> list[float]:
    p = k / n
    h = 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n)
    return [round(100 * p, 3), round(100 * h, 3)]


def agg_to_json(agg: Agg) -> dict:
    n = agg.n
    out: dict = {"n": n, "persons": {}}
    for pid in agg.pids:
        ever_exec = sum(v for r, v in agg.max_rank[pid].items() if r >= 5)
        nx = max(1, agg.promo_n_exec[pid])
        ns = max(1, agg.staff_n[pid])
        out["persons"][pid] = {
            "final": {c: ci(v, n) for c, v in agg.final[pid].most_common()},
            "ever_exec": ci(ever_exec, n),
            "ever_toudori": ci(agg.ever_toudori.get(pid, 0), n),
            "max_rank": dict(sorted(agg.max_rank[pid].items())),
            "setback_runs": {k: ci(v, n) for k, v in agg.setback_runs[pid].most_common()},
            "setback_events": dict(agg.setback[pid].most_common()),
            "sanction_level": dict(agg.sanction_level[pid]),
            "secret_origin": dict(agg.secret_origin[pid].most_common()),
            "promotions": agg.promo_n[pid] / n,
            "exec_promo_margin": {k: round(v / nx, 3) for k, v in agg.promo_comp[pid].items()},
            "n_exec_promo": agg.promo_n_exec[pid],
            "staff_promo_components": {k: round(v / ns, 3) for k, v in agg.staff_comp[pid].items()},
            "n_staff_promo": agg.staff_n[pid],
            "actions": {
                k: round(v / max(1, sum(agg.actions[pid].values())), 4)
                for k, v in agg.actions[pid].most_common()
            },
        }
    out["final_toudori"] = {k or "不在": ci(v, n) for k, v in agg.final_toudori.most_common()}
    years: dict = {}
    for (y, who), v in agg.toudori_year.items():
        years.setdefault(str(y), {})[who] = round(100 * v / n, 2)
    out["toudori_by_year"] = years
    out["edges"] = [[a, b, lv, v] for (a, b, lv), v in agg.edges.most_common(60)]
    return out


def paired(base: Agg, other: Agg, cats: dict[str, int]) -> dict:
    """同じ試行番号どうしの差。役員経験・10年後のカテゴリの割合（ポイント）と、その95%信頼区間の半幅。"""
    ids = sorted(set(base.per_run) & set(other.per_run))
    n = len(ids)
    inv = {v: k for k, v in cats.items()}
    bc = np.stack([base.per_run[i][0] for i in ids])
    oc = np.stack([other.per_run[i][0] for i in ids])
    bm = np.stack([base.per_run[i][1] for i in ids])
    om = np.stack([other.per_run[i][1] for i in ids])
    out = {"n": n, "persons": {}}
    for j, pid in enumerate(base.pids):
        d = (om[:, j] >= 5).astype(float) - (bm[:, j] >= 5).astype(float)
        row: dict[str, Any] = {
            "ever_exec": [round(100 * d.mean(), 3), round(196 * d.std(ddof=1) / math.sqrt(n), 3)]
        }
        changed = float((bc[:, j] != oc[:, j]).mean())
        row["changed_runs"] = round(100 * changed, 3)
        for c in sorted(set(bc[:, j]) | set(oc[:, j])):
            dc = (oc[:, j] == c).astype(float) - (bc[:, j] == c).astype(float)
            if abs(dc.mean()) >= 0.001:
                row[inv[int(c)]] = [round(100 * dc.mean(), 3), round(196 * dc.std(ddof=1) / math.sqrt(n), 3)]
        out["persons"][pid] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baseline", "interventions", "sensitivity"])
    ap.add_argument("--policy", default="sonnet")
    ap.add_argument("--runs", type=int, default=10000)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--only", default=None, help="介入・パラメータの名前をカンマ区切りで限定する")
    args = ap.parse_args()
    out_dir = ROOT / "data" / "results" / args.policy
    out_dir.mkdir(parents=True, exist_ok=True)
    pol = policies_for(args.policy)
    t0 = time.time()

    base, cats = collect(args.runs, SEED, Params(), pol, processes=args.procs)
    print(f"基準 {args.runs:,} 回 {time.time() - t0:.0f}秒", flush=True)
    if args.cmd == "baseline":
        (out_dir / f"baseline_{args.runs}.json").write_text(
            json.dumps(agg_to_json(base), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return

    results = {}
    if args.cmd == "interventions":
        names = args.only.split(",") if args.only else list(INTERVENTIONS)
        for name in names:
            t1 = time.time()
            other, _ = collect(args.runs, SEED, Params(), pol, INTERVENTIONS[name], processes=args.procs)
            results[name] = {"paired": paired(base, other, cats), "summary": agg_to_json(other)}
            print(f"  {name} {time.time() - t1:.0f}秒", flush=True)
    else:
        names = args.only.split(",") if args.only else list(SENSITIVITY)
        for name in names:
            mode, lo, hi = SENSITIVITY[name]
            base_v = getattr(Params(), name)
            for tag, f in (("low", lo), ("high", hi)):
                v = base_v * f if mode == "mul" else base_v + f
                p = dataclasses.replace(Params(), **{name: v})
                t1 = time.time()
                other, _ = collect(args.runs, SEED, p, pol, processes=args.procs)
                results[f"{name}:{tag}"] = {"value": v, "paired": paired(base, other, cats)}
                print(f"  {name}={v:.4g} {time.time() - t1:.0f}秒", flush=True)
    (out_dir / f"{args.cmd}_{args.runs}.json").write_text(
        json.dumps({"base": agg_to_json(base), "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
