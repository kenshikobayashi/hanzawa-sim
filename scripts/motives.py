"""LLM が答えた行動の理由を、語のパターンで動機に分類する（PLAN.md 5章「行動の理由」）。

1つの理由が複数の動機に当てはまることがある。分類は語の一致による簡易なもので、文脈は見ない。
  uv run python scripts/motives.py --model sonnet
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from hanzawa_sim.agents.distill import load_answers
from hanzawa_sim.world.loader import load_scenario

ROOT = Path(__file__).resolve().parents[1]
MOTIVES = {
    "報復・追及": r"報い|倍返し|許さ|許せ|追及|仇|けじめ|やり返|屈辱|執着|反撃",
    "保身": r"保身|身を守|露見|弱み|火種|隠蔽|守りを固|自滅|傷を避",
    "組織・融和": r"銀行のため|銀行の安定|銀行の信用|組織|融和|行内の安定|体質",
    "出世・地位": r"昇進|出世|ポスト|返り咲|上を目指|復権|上の席|地位|昇格|頭取の座|頭取を目指",
    "仲間・部下": r"仲間|部下|同期|味方|恩",
    "業務・成果": r"業績|結果を出|本業|顧客|成果|実績",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()
    names = {p["id"]: p["name"] for p in load_scenario().persons}
    ans = [
        a
        for a in load_answers(ROOT / "data" / "elicitation" / "main")
        if a.model == args.model and a.schema == "action" and a.variant == "named" and not a.retest
    ]
    count: dict[str, Counter] = defaultdict(Counter)
    n: Counter = Counter()
    for a in ans:
        n[a.pid] += 1
        for m, pat in MOTIVES.items():
            if re.search(pat, a.reason):
                count[a.pid][m] += 1
    out = {pid: {m: round(100 * count[pid][m] / n[pid], 1) for m in MOTIVES} for pid in n}
    path = ROOT / "data" / "results" / args.model / "motives.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("| 人物 | 回答数 | " + " | ".join(MOTIVES) + " |")
    print("|---" * (len(MOTIVES) + 2) + "|")
    for pid, name in names.items():
        if pid in out:
            print(f"| {name} | {n[pid]} | " + " | ".join(f"{out[pid][m]:.0f}" for m in MOTIVES) + " |")


if __name__ == "__main__":
    main()
