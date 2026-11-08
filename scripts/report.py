"""結果の JSON から Markdown の表を作る（GitHub・note 用）。

  uv run python scripts/report.py --policy sonnet --runs 100000 --iv-runs 10000 --sens-runs 5000
出力: data/results/<policy>/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hanzawa_sim.world.loader import load_scenario

ROOT = Path(__file__).resolve().parents[1]
CATS = [
    "頭取",
    "副頭取・専務・常務",
    "取締役・執行役員",
    "部長",
    "次長以下",
    "出向",
    "転籍",
    "退職・退任",
    "失脚",
]
ACT_JA = {
    "WORK": "業務", "LIE_LOW": "静観", "COVER_UP": "隠蔽", "MISCONDUCT": "不正", "RESIGN": "退職", "GATHER": "証拠収集",
    "EXPOSE": "告発", "BARGAIN": "取引", "UNDERMINE": "妨害", "LOBBY": "根回し", "SUPPORT": "支援",
}  # fmt: skip
COMP_JA = {
    "perf": "業績", "ability": "能力", "sponsor": "後ろ盾", "relation": "頭取・役員との関係", "faction": "派閥",
    "age": "年齢", "scandal": "処分歴", "reputation": "評判", "decider": "頭取の好み", "base": "基準",
}  # fmt: skip


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(v) -> str:
    return f"{v[0]:.1f}" if isinstance(v, list) else f"{v:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="sonnet")
    ap.add_argument("--runs", type=int, default=100000)
    ap.add_argument("--iv-runs", type=int, default=10000)
    ap.add_argument("--sens-runs", type=int, default=5000)
    args = ap.parse_args()
    d = ROOT / "data" / "results" / args.policy
    sc = load_scenario()
    name = {p["id"]: p["name"] for p in sc.persons}

    def nm(pid: str) -> str:
        return name.get(pid, pid)

    base = load(d / f"baseline_{args.runs}.json")
    out: list[str] = [f"# 結果（方策: {args.policy}）", ""]
    if base:
        n = base["n"]
        out += [f"## 10年後（2023年9月）の状態の分布 [%]（{n:,} 回）", ""]
        out.append("| 人物 | " + " | ".join(CATS) + " | 役員経験 | 頭取経験 |")
        out.append("|---" * (len(CATS) + 3) + "|")
        for pid, r in base["persons"].items():
            if pid == "kurosaki":
                continue
            row = [fmt(r["final"].get(c, [0.0])) for c in CATS]
            out.append(
                f"| {nm(pid)} | " + " | ".join(row) + f" | {fmt(r['ever_exec'])} | {fmt(r['ever_toudori'])} |"
            )
        k = base["persons"].get("kurosaki")
        if k:
            out += ["", "黒崎駿一（金融庁）: " + "、".join(f"{c} {v[0]:.1f}%" for c, v in k["final"].items())]
        out += ["", "## 10年後の頭取 [%]", ""]
        out += [f"- {nm(k_)}: {v[0]:.1f}" for k_, v in list(base["final_toudori"].items())[:8]]
        out += [
            "",
            "## 頭取の推移（毎年9月末、上位3人）[%]",
            "",
            "| 年 | 1位 | 2位 | 3位 |",
            "|---|---|---|---|",
        ]
        for y, who in sorted(base["toudori_by_year"].items()):
            top = sorted(who.items(), key=lambda kv: -kv[1])[:3]
            out.append(f"| {y} | " + " | ".join(f"{nm(k_)} {v:.1f}" for k_, v in top) + " |")
        out += [
            "",
            "## 後退（降格・出向・失脚・退職）の引き金（その引き金が1回以上起きた試行の割合 [%]、上位5）",
            "",
        ]
        for pid, r in base["persons"].items():
            items = list(r["setback_runs"].items())[:5]
            if items:
                txt = "、".join(
                    f"{k_.replace('（', '（').replace(pid, nm(pid))} {v[0]:.1f}" for k_, v in items
                )
                for p2 in name:
                    txt = txt.replace(f"（{p2}）", f"（{nm(p2)}）")
                out.append(f"- {nm(pid)}: {txt}")
        out += ["", "## 役員への昇進の要因（勝者と次点の候補のスコアの差の平均。正の値ほど勝因）", ""]
        keys = sorted({k_ for r in base["persons"].values() for k_ in r["exec_promo_margin"]})
        out.append("| 人物 | 件数/試行 | " + " | ".join(COMP_JA.get(k_, k_) for k_ in keys) + " |")
        out.append("|---" * (len(keys) + 2) + "|")
        for pid, r in base["persons"].items():
            if r["n_exec_promo"] >= max(20, n // 1000):
                m = r["exec_promo_margin"]
                out.append(
                    f"| {nm(pid)} | {r['n_exec_promo'] / n:.3f} | "
                    + " | ".join(f"{m.get(k_, 0.0):+.2f}" for k_ in keys)
                    + " |"
                )
        out += ["", "## 誰が誰の処分の原因になったか（上位15。件数/1,000試行）", ""]
        for a, b, lv, v in base["edges"][:15]:
            out.append(f"- {nm(a)} → {nm(b)}（{lv}）: {1000 * v / n:.1f}")
        out += ["", "## 行動の頻度 [%]", ""]
        acts = list(ACT_JA)
        out.append("| 人物 | " + " | ".join(ACT_JA[a] for a in acts) + " |")
        out.append("|---" * (len(acts) + 1) + "|")
        for pid, r in base["persons"].items():
            out.append(
                f"| {nm(pid)} | " + " | ".join(f"{100 * r['actions'].get(a, 0):.0f}" for a in acts) + " |"
            )

    iv = load(d / f"interventions_{args.iv_runs}.json")
    if iv:
        focus = ["hanzawa", "owada", "tomari", "kondo", "naito", "kurosaki", "asano", "takagi"]
        out += ["", f"## 介入実験（共通乱数による対の比較、{args.iv_runs:,} 組）", "",
                "役員経験の確率の変化 [ポイント]（±は95%信頼区間の半幅）。黒崎は金融庁の課長以上への到達", ""]  # fmt: skip
        out.append("| 介入 | " + " | ".join(nm(p) for p in focus) + " |")
        out.append("|---" * (len(focus) + 1) + "|")
        for k_, v in iv["results"].items():
            ps = v["paired"]["persons"]
            out.append(
                f"| {k_} | "
                + " | ".join(f"{ps[p]['ever_exec'][0]:+.1f}±{ps[p]['ever_exec'][1]:.1f}" for p in focus)
                + " |"
            )
    sens = load(d / f"sensitivity_{args.sens_runs}.json")
    if sens:
        focus = ["hanzawa", "owada", "tomari", "kondo", "naito"]
        out += [
            "",
            f"## 感度分析（パラメータを 0.5倍・1.5倍にしたときの役員経験の確率の変化 [ポイント]、{args.sens_runs:,} 組）",
            "",
        ]
        out.append("| パラメータ | " + " | ".join(nm(p) for p in focus) + " |")
        out.append("|---" * (len(focus) + 1) + "|")
        for k_, v in sens["results"].items():
            ps = v["paired"]["persons"]
            out.append(f"| {k_} | " + " | ".join(f"{ps[p]['ever_exec'][0]:+.1f}" for p in focus) + " |")
    rep = load(ROOT / "data" / "policies" / f"report_{args.policy}.json")
    if rep:
        a = rep["action"]
        out += ["", "## 方策モデルの誤差（全変動距離 TV、0〜1）", "",
                f"- 検証用カードでの方策モデルと LLM の回答の差: {a['val_tv_distilled']:.3f}",
                f"- 比較: 人物ごとの平均の分布を返すだけのモデル: {a['val_tv_person_mean']:.3f}",
                f"- 比較: LLM に同じカードへもう一度答えさせた差: {a['llm_retest_tv']:.3f}（{a['n_retest']} 組）",
                f"- 人物名・組織名を伏せた版と元の版の差: {a['anon_vs_named_tv']:.3f}（同じ人物の再回答の差 {a['anon_ref_retest_tv']:.3f}）",
                f"- 再現テスト（2013年版の場面）: {sum(v['hit'] for v in rep['repro'].values())}/{len(rep['repro'])} 場面で劇中の行動が最も高い確率"]  # fmt: skip
    (d / "report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{d / 'report.md'}（{len(out)} 行）")


if __name__ == "__main__":
    main()
