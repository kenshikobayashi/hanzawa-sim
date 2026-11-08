"""Phase 4: LLM エージェントの回答から方策モデルを学習し、誤差を評価する。

  uv run python scripts/fit_policies.py            # sonnet と haiku の両方
  uv run python scripts/fit_policies.py --model sonnet

出力:
  data/policies/{model}.json         方策モデル（engine で読み込む）
  data/policies/report_{model}.json  誤差の評価、再現テスト、情報漏れの検査
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from hanzawa_sim.agents.distill import Answer, cv_fit, evaluate, load_answers, person_mean_baseline, tv
from hanzawa_sim.agents.elicit import RESPONSE_KEYS, SANCTION_KEYS
from hanzawa_sim.agents.features import FEATURES
from hanzawa_sim.agents.persona import ACTION_KEYS
from hanzawa_sim.agents.special_cards import (
    HR_FEATURES,
    HR_KEYS,
    LEAK_WORDS,
    REPRO,
    RESPONSE_FEATURES,
    SANCTION_FEATURES,
)
from hanzawa_sim.world.model import Action

ROOT = Path(__file__).resolve().parents[1]
ELICIT = ROOT / "data" / "elicitation" / "main"
OUT = ROOT / "data" / "policies"
ACTIONS = [ACTION_KEYS[a] for a in Action]


def pair_tv(a: list[Answer], b: list[Answer]) -> tuple[float, int]:
    """同じカードへの2組の回答の平均 TV（選べる行動が同じもの）。"""
    idx = {(x.pid, x.card_id): x for x in b}
    ds = [tv(x.p, idx[(x.pid, x.card_id)].p) for x in a if (x.pid, x.card_id) in idx]
    return (float(np.mean(ds)) if ds else float("nan")), len(ds)


def probes(model: str) -> dict:
    jobs = {j["id"]: j for j in json.loads((ELICIT / "jobs.json").read_text(encoding="utf-8"))}
    out = {}
    for line in (ELICIT / "calls.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        j = jobs.get(r["tag"])
        if j and j["schema"] == "probe" and j["model"] == model and r.get("output"):
            preds = r["output"].get("predictions", [])
            out[j["pid"]] = {
                "predictions": preds,
                "leak_words": sorted({w for w in LEAK_WORDS for p in preds if w in p}),
            }
    return out


def fit_model(model: str) -> dict:
    ans = [a for a in load_answers(ELICIT) if a.model == model]
    named = [a for a in ans if a.variant == "named" and not a.retest]
    act = [a for a in named if a.schema == "action"]
    train = [a for a in act if a.split == "train"]
    val = [a for a in act if a.split == "val"]
    print(f"[{model}] 行動カード 学習 {len(train)} 検証 {len(val)}")

    act_model, act_cv = cv_fit(
        train, list(FEATURES), ACTIONS, (0.0001, 0.0003, 0.001, 0.003), (0.0003, 0.001, 0.003, 0.01)
    )
    by_pid = defaultdict(list)
    for a in val:
        by_pid[a.pid].append(a)
    val_tv = {pid: round(evaluate(act_model, xs), 3) for pid, xs in sorted(by_pid.items())}

    retest = [a for a in ans if a.retest and a.schema == "action"]
    anon = [a for a in ans if a.variant == "anon"]
    retest_tv, n_retest = pair_tv(retest, act)
    anon_tv, n_anon = pair_tv(anon, act)
    anon_ref, _ = pair_tv([a for a in retest if a.pid in {x.pid for x in anon}], act)

    resp = [a for a in named if a.schema == "response"]
    sanc = [a for a in named if a.schema == "sanction"]
    hr = [a for a in named if a.schema == "hr"]

    def fit_or_none(data, feats, keys):
        if len(data) < 10:  # 途中の結果で試すとき
            return None, {"best": None}
        return cv_fit(data, list(feats), keys, (0.001, 0.003, 0.01, 0.03), (0.01, 0.03, 0.1, 0.3, 1.0))

    resp_model, resp_cv = fit_or_none(resp, RESPONSE_FEATURES, RESPONSE_KEYS)
    sanc_model, sanc_cv = fit_or_none(sanc, SANCTION_FEATURES, SANCTION_KEYS)
    hr_model, hr_cv = fit_or_none(hr, HR_FEATURES, HR_KEYS)

    # 再現テスト: 劇中の行動に割り当てた確率と、それが最も高い確率か
    repro = {}
    for a in ans:
        if a.schema.startswith("repro"):
            canon = next(c for pid, _, c, _ in REPRO if pid == a.pid)
            keys = {
                "repro-action": ACTIONS,
                "repro-response": RESPONSE_KEYS,
                "repro-sanction": SANCTION_KEYS,
            }[a.schema]
            pc = sum(a.p[keys.index(k)] for k in canon)
            repro[a.pid] = {
                "canon": canon,
                "p_canon": round(float(pc), 3),
                "top": keys[int(np.argmax(a.p))],
                "hit": keys[int(np.argmax(a.p))] in canon,
                "reason": a.reason,
            }

    policy = {
        "model": model,
        "action": act_model.to_json(),
        "response": resp_model.to_json() if resp_model else None,
        "sanction": sanc_model.to_json() if sanc_model else None,
        "hr": hr_model.to_json() if hr_model else None,
    }
    report = {
        "model": model,
        "n": {
            "train": len(train),
            "val": len(val),
            "response": len(resp),
            "sanction": len(sanc),
            "hr": len(hr),
        },
        "action": {
            "val_tv_distilled": round(evaluate(act_model, val), 4),
            "val_tv_person_mean": round(person_mean_baseline(train, val), 4),
            "train_tv_distilled": round(evaluate(act_model, train), 4),
            "llm_retest_tv": round(retest_tv, 4),
            "n_retest": n_retest,
            "anon_vs_named_tv": round(anon_tv, 4),
            "anon_ref_retest_tv": round(anon_ref, 4),
            "n_anon": n_anon,
            "val_tv_by_person": val_tv,
            "mean_other_mass": round(float(np.mean([a.other_mass for a in act])), 4),
            "mean_masked_mass": round(float(np.mean([a.masked_mass for a in act])), 4),
            "cv": act_cv,
        },
        "response": {"cv": resp_cv, "train_tv": round(evaluate(resp_model, resp), 4) if resp_model else None},
        "sanction": {"cv": sanc_cv, "train_tv": round(evaluate(sanc_model, sanc), 4) if sanc_model else None},
        "hr": {"cv": hr_cv, "train_tv": round(evaluate(hr_model, hr), 4) if hr_model else None},
        "repro": repro,
        "probe": probes(model),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model}.json").write_text(json.dumps(policy, ensure_ascii=False), encoding="utf-8")
    (OUT / f"report_{model}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sonnet", "haiku"], default=None)
    args = ap.parse_args()
    reports = {}
    for m in [args.model] if args.model else ["sonnet", "haiku"]:
        r = fit_model(m)
        reports[m] = r
        a = r["action"]
        print(
            f"  行動: 検証TV 方策モデル {a['val_tv_distilled']:.3f} / 人物平均 {a['val_tv_person_mean']:.3f} / "
            f"LLM再回答 {a['llm_retest_tv']:.3f}（{a['n_retest']}組）/ 名前を伏せた版 {a['anon_vs_named_tv']:.3f}"
            f"（同じ人物の再回答 {a['anon_ref_retest_tv']:.3f}）"
        )
        print(f"  正則化 {a['cv']['best']}  人物別の検証TV {a['val_tv_by_person']}")
        for k in ("response", "sanction", "hr"):
            if r[k]["train_tv"] is None:
                continue
            print(
                f"  {k}: 学習TV {r[k]['train_tv']:.3f} 交差検証 {min(v for kk, v in r[k]['cv'].items() if kk != 'best'):.3f}"
            )
        hits = [v["hit"] for v in r["repro"].values()]
        print(
            f"  再現テスト: {sum(hits)}/{len(hits)} 一致",
            {k: (v["top"], v["p_canon"]) for k, v in r["repro"].items()},
        )
        print(
            "  情報漏れの語:",
            {k: v["leak_words"] for k, v in r["probe"].items() if v["leak_words"]} or "なし",
        )
    if len(reports) == 2:
        ans = load_answers(ELICIT)
        s = [
            a
            for a in ans
            if a.model == "sonnet" and a.variant == "named" and not a.retest and a.schema == "action"
        ]
        h = [
            a
            for a in ans
            if a.model == "haiku" and a.variant == "named" and not a.retest and a.schema == "action"
        ]
        gap, n = pair_tv(s, h)
        print(f"Sonnet と Haiku の同じカードへの回答の差: TV {gap:.3f}（{n}組）")
        (OUT / "model_gap.json").write_text(json.dumps({"tv": gap, "n": n}), encoding="utf-8")


if __name__ == "__main__":
    main()
