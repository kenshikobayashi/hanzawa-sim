"""Phase 2 パイロット（LLM呼び出し 18回）。

目的:
  1. 呼び出し1回あたりのトークン数・所要時間・利用枠の消費を測り、本番で1枠に何回呼べるかを見積もる
  2. Haiku と Sonnet の回答の差、5枚まとめて答えさせた場合と1枚ずつの差、同じカードの再回答の差を測る
  3. 取引への応答・処分の判断の形式を試す
  4. 2013年以降の出来事（2020年版の内容）を答えてしまうか（学習データからの情報漏れ）を確かめる

利用枠の消費を測るため、実行の前後に Claude Code で /usage を確認してください。
  uv run python scripts/pilot.py --dry-run   # 呼び出さずにプロンプトとカードを表示
  uv run python scripts/pilot.py             # 実行（約5分）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hanzawa_sim.agents.cards import Card, sample_cards
from hanzawa_sim.agents.elicit import (
    ACTION_SCHEMA,
    PROBE_SCHEMA,
    RESPONSE_SCHEMA,
    SANCTION_SCHEMA,
    batch_prompt,
    normalize_action,
    tv_distance,
)
from hanzawa_sim.agents.llm import ClaudeCLI, LLMResult, UsageLimitError
from hanzawa_sim.agents.persona import load_personas, system_prompt
from hanzawa_sim.world.loader import load_scenario

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "elicitation" / "pilot"
MODELS = ("haiku", "sonnet")
LEAK_WORDS = (
    "電脳",
    "スパイラル",
    "帝国航空",
    "伊佐山",
    "森山",
    "瀬名",
    "三笠",
    "紀本",
    "箕部",
    "白井",
    "曾根崎",
    "乃原",
    "牧野",
)

RESPONSE_CARD = """【状況 R1】2015年4月時点
- あなたの立場: 東京中央銀行 調査役（本部、45歳）
- 取締役の大和田暁から次の申し出を受けた:「あなたが入手した、半沢直樹の出向先での規則違反を示す資料を私に渡せば、希望する部署の次長に昇格させる」
- 資料が大和田に渡れば、半沢は処分を受けるおそれがある
- 選べる応答: accept（申し出を受けて資料を渡す）, reject（断る）, retaliate（申し出のことを半沢や上層部に知らせ、大和田に対抗する）"""

SANCTION_CARD = """【状況 J1】2016年4月時点
- あなたは頭取として処分を決める立場にある
- 旧S出身の常務（大和田暁ではない別の人物。有能で、旧S派の有力者）が、取引先への不正な融資に関与していたことが、確かな証拠とともに役員会議で明らかになった。不正の重大さは中程度より重い
- 処分の選択肢: warning（注意のみ）, demote（降格して銀行に残す）, second（子会社・取引先へ出向させる）, oust（解任・懲戒）"""

PROBE = "2013年9月の今、あなたはこれから1〜2年の間に自分の身に何が起きると予想しますか。具体的に3つ挙げてください。"


def tokens(r: LLMResult) -> tuple[int, int]:
    u = r.usage or {}
    tin = (
        int(u.get("input_tokens", 0))
        + int(u.get("cache_creation_input_tokens", 0))
        + int(u.get("cache_read_input_tokens", 0))
    )
    return tin, int(u.get("output_tokens", 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sc = load_scenario()
    personas = load_personas()
    pids = ["hanzawa", "owada", "nakanowatari"]
    cards = sample_cards(sc, pids, n_per_person=5, seed=101)
    sp = {pid: system_prompt(sc, pid, personas) for pid in pids + ["kondo"]}

    if args.dry_run:
        print(sp["hanzawa"])
        print("=" * 60)
        print(batch_prompt(cards["hanzawa"]))
        print("=" * 60)
        print({pid: len(sp[pid]) for pid in sp}, "文字（システムプロンプト）")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    be = {m: ClaudeCLI(m, OUT / "calls.jsonl", OUT / "prompts") for m in MODELS}
    calls: list[dict] = []
    results: dict[str, dict] = {}

    def call(tag: str, model: str, pid: str, prompt: str, schema: dict) -> LLMResult:
        print(f"  {tag:<28} {model:<6}", end=" ", flush=True)
        r = be[model].call(tag, sp[pid], prompt, schema)
        tin, tout = tokens(r)
        calls.append(
            {
                "tag": tag,
                "model": model,
                "ok": r.ok,
                "in": tin,
                "out": tout,
                "cost": r.cost_usd,
                "ms": r.duration_ms,
            }
        )
        print(f"ok={r.ok} in={tin} out={tout} {r.duration_ms / 1000:.1f}s")
        return r

    def answers_by_card(r: LLMResult, cs: list[Card]) -> dict[str, dict]:
        out = {}
        if not r.ok or not r.data:
            return out
        masks = {c.card_id: c.mask for c in cs}
        for a in r.data.get("answers", []):
            if a.get("card_id") in masks:
                out[a["card_id"]] = {
                    **normalize_action(a, masks[a["card_id"]]),
                    "reason": a.get("reason", ""),
                    "other_action": a.get("other_action", ""),
                }
        return out

    t0 = time.time()
    try:
        print("A. 5枚まとめて回答（3人 × 2モデル）")
        for pid in pids:
            for m in MODELS:
                results[f"A/{pid}/{m}"] = answers_by_card(
                    call(f"A/{pid}", m, pid, batch_prompt(cards[pid]), ACTION_SCHEMA), cards[pid]
                )
        print("B. 1枚ずつ回答（半沢 × 5枚、haiku）")
        single = {}
        for c in cards["hanzawa"]:
            single.update(
                answers_by_card(
                    call(f"B/{c.card_id}", "haiku", "hanzawa", batch_prompt([c]), ACTION_SCHEMA), [c]
                )
            )
        results["B/hanzawa/haiku"] = single
        print("C. 同じ5枚の再回答（半沢、haiku）")
        results["C/hanzawa/haiku"] = answers_by_card(
            call("C/hanzawa", "haiku", "hanzawa", batch_prompt(cards["hanzawa"]), ACTION_SCHEMA),
            cards["hanzawa"],
        )
        print("D. 取引への応答（近藤）、E. 処分の判断（中野渡）、F. 情報漏れの検査（半沢）")
        for m in MODELS:
            r = call(
                "D/kondo",
                m,
                "kondo",
                "以下の状況で、各応答を選ぶ確率（合計100）と理由を答えてください。\n\n" + RESPONSE_CARD,
                RESPONSE_SCHEMA,
            )
            results[f"D/kondo/{m}"] = r.data or {}
            r = call(
                "E/nakanowatari",
                m,
                "nakanowatari",
                "以下の状況で、各処分を選ぶ確率（合計100）と理由を答えてください。\n\n" + SANCTION_CARD,
                SANCTION_SCHEMA,
            )
            results[f"E/nakanowatari/{m}"] = r.data or {}
            r = call("F/hanzawa", m, "hanzawa", PROBE, PROBE_SCHEMA)
            preds = (r.data or {}).get("predictions", [])
            results[f"F/hanzawa/{m}"] = {
                "predictions": preds,
                "leak_words": sorted({w for w in LEAK_WORDS for p in preds if w in p}),
            }
    except UsageLimitError as e:
        print(f"\n利用枠の上限に達したため中断: {e}")

    # ---- 比較
    def mean_tv(a: dict, b: dict) -> float | None:
        ks = [k for k in a if k in b and a[k]["dist"] and b[k]["dist"]]
        return round(sum(tv_distance(a[k]["dist"], b[k]["dist"]) for k in ks) / len(ks), 3) if ks else None

    comp = {
        "haiku_vs_sonnet": {
            pid: mean_tv(results.get(f"A/{pid}/haiku", {}), results.get(f"A/{pid}/sonnet", {}))
            for pid in pids
        },
        "batch_vs_single(haiku,hanzawa)": mean_tv(
            results.get("A/hanzawa/haiku", {}), results.get("B/hanzawa/haiku", {})
        ),
        "retest(haiku,hanzawa)": mean_tv(
            results.get("A/hanzawa/haiku", {}), results.get("C/hanzawa/haiku", {})
        ),
    }
    totals = {}
    for m in MODELS:
        cs = [c for c in calls if c["model"] == m]
        if cs:
            totals[m] = {
                "calls": len(cs),
                "ok": sum(c["ok"] for c in cs),
                "in": sum(c["in"] for c in cs),
                "out": sum(c["out"] for c in cs),
                "cost_usd": round(sum(c["cost"] for c in cs), 4),
                "mean_s": round(sum(c["ms"] for c in cs) / len(cs) / 1000, 1),
            }
    report = {
        "elapsed_s": round(time.time() - t0),
        "totals": totals,
        "comparison": comp,
        "calls": calls,
        "results": results,
        "cards": {pid: [c.__dict__ for c in cards[pid]] for pid in pids},
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n=== 集計 ===")
    print(json.dumps({"totals": totals, "comparison": comp}, ensure_ascii=False, indent=1))
    for m in MODELS:
        for key in (f"D/kondo/{m}", f"E/nakanowatari/{m}"):
            ans = (results.get(key) or {}).get("answers", [{}])
            print(key, ans[0].get("probs") if ans else None)
        print(f"F/hanzawa/{m}", results.get(f"F/hanzawa/{m}", {}).get("leak_words"))
    print(f"\n詳細: {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
