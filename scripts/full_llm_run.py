"""完全LLMラン: 名前付きの人物全員の行動を、毎半期 LLM エージェントが決める10年分の試行（1回あたり約250回の呼び出し）。

  uv run python scripts/full_llm_run.py --run-id 0 --model sonnet --policy sonnet

出力（data/fullrun/<model>-<run_id>/）:
  calls.jsonl     全呼び出しの記録
  records.json    半期ごと・人物ごとの状況カード、LLM の回答、方策モデルの予測、選ばれた行動
  timeline.md     試行の経過（人事・告発・処分・取引）
  summary.json    10年後の状態と、試行中に現れた状態での LLM と方策モデルの差
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hanzawa_sim.agents.distill import tv
from hanzawa_sim.agents.llm import ClaudeCLI
from hanzawa_sim.agents.llm_run import LEVEL_JA, MOVE_REASON, LLMRun, _pos
from hanzawa_sim.agents.persona import load_personas, system_prompt
from hanzawa_sim.agents.policy import load_policy_set
from hanzawa_sim.sim.runner import final_category
from hanzawa_sim.world.loader import load_scenario
from hanzawa_sim.world.params import Params

ROOT = Path(__file__).resolve().parents[1]
SEED = 20131001  # analyze.py と同じ。同じ試行番号の方策モデルの試行と比べられる


def timeline(run: LLMRun) -> str:
    w = run.w
    nm = lambda i: w.persons[i].name if w.persons[i].named else f"旧{w.persons[i].faction}出身の行員"
    lines = []
    for e in run.events:
        t = run.p.start_time + 0.5 * e.step
        ym = f"{int(t)}年{10 if t - int(t) >= 0.5 else 4}月"
        if e.etype == "MOVE" and w.persons[e.target].named:
            d = e.detail or {}
            q = w.persons[e.target]
            frm, to = (
                _pos(int(q.org), d["from_rank"], d["from_track"]),
                _pos(int(q.org), d["to_rank"], d["to_track"]),
            )
            lines.append(f"- {ym} {nm(e.target)}: {frm} → {to}（{MOVE_REASON.get(e.outcome, e.outcome)}）")
        elif e.etype == "EXPOSE" and (w.persons[e.actor].named or w.persons[e.target].named):
            res = "成功" if e.outcome == "success" else "失敗"
            lines.append(f"- {ym} {nm(e.actor)}が{nm(e.target)}の不正を告発（{res}）")
        elif e.etype == "EXPOSED" and w.persons[e.target].named and e.outcome != "expose":
            via = {"inspection": "金融庁検査", "leak": "内部からの露見", "intervention": "介入"}.get(
                e.outcome, e.outcome
            )
            lines.append(f"- {ym} {nm(e.target)}の不正が{via}で発覚")
        elif e.etype == "SANCTION" and w.persons[e.target].named:
            kind = (e.detail or {}).get("kind", "")
            lines.append(f"- {ym} {nm(e.target)}に処分: {LEVEL_JA.get(e.outcome, e.outcome)}（{kind}）")
        elif e.etype == "BARGAIN" and (w.persons[e.actor].named or w.persons[e.target].named):
            kind = "脅し" if (e.detail or {}).get("kind") == "threat" else "懐柔"
            r = {"ACCEPT": "応じた", "REJECT": "断った", "RETALIATE": "対抗した"}[e.outcome]
            lines.append(f"- {ym} {nm(e.actor)}が{nm(e.target)}に取引（{kind}）→ {r}")
        elif e.etype in ("CASE_FAILURE", "LATENT_SCANDAL") and w.persons[e.target].named:
            what = "大型案件の失敗" if e.etype == "CASE_FAILURE" else "派閥の過去の不祥事の責任"
            lines.append(f"- {ym} {nm(e.target)}: {what}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, default=0)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--policy", default="sonnet")
    args = ap.parse_args()
    out = ROOT / "data" / "fullrun" / f"{args.model}-{args.run_id}"
    out.mkdir(parents=True, exist_ok=True)

    sc = load_scenario()
    pids = [p["id"] for p in sc.persons]
    personas = load_personas()
    prompts = {pid: system_prompt(sc, pid, personas) for pid in pids}
    distilled = load_policy_set(ROOT / "data" / "policies" / f"{args.policy}.json", pids)
    backend = ClaudeCLI(args.model, out / "calls.jsonl", ROOT / "data" / "elicitation" / "main" / "prompts")
    run = LLMRun(
        sc, Params(), distilled, SEED, args.run_id, backend=backend, prompts=prompts, distilled=distilled
    )
    t0 = time.time()
    for t in range(run.p.n_steps):
        run.step(t)
        n = sum(r["step"] == t for r in run.records)
        print(f"  半期 {t + 1}/20（{n}人）{(time.time() - t0) / 60:.1f}分", flush=True)

    recs = run.records
    tvs = [tv(np.array(r["llm"]), np.array(r["distilled"])) for r in recs if r["llm"] and r["distilled"]]
    summ = run.summary()
    summary = {
        "run_id": args.run_id,
        "model": args.model,
        "minutes": round((time.time() - t0) / 60, 1),
        "n_decisions": len(recs),
        "llm_failures": sum(r["source"] != "llm" for r in recs),
        "on_policy_tv_mean": float(np.mean(tvs)) if tvs else None,
        "on_policy_tv_by_person": {
            pid: round(
                float(
                    np.mean(
                        [
                            tv(np.array(r["llm"]), np.array(r["distilled"]))
                            for r in recs
                            if r["pid"] == pid and r["llm"] and r["distilled"]
                        ]
                    )
                ),
                3,
            )
            for pid in pids
            if any(r["pid"] == pid and r["llm"] for r in recs)
        },
        "final": {pid: final_category(s) for pid, s in summ.items()},
        "max_rank": {pid: s["max_rank"] for pid, s in summ.items()},
        "toudori": run.toudori_hist,
    }
    (out / "records.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "timeline.md").write_text(timeline(run), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_decisions", "llm_failures", "on_policy_tv_mean", "final")},
                     ensure_ascii=False, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
