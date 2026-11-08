"""解説ページ（docs/method.md、docs/results.md）の図を作る。

各図を明るい背景用（<名前>.png）と暗い背景用（<名前>-dark.png）の2枚で出力する。
配色は、色覚の多様性への配慮を検証済みのカテゴリ配色（8色、固定の順序）、単一色相の連続配色、
青と赤の発散配色を使う。
  uv run python scripts/figures.py
出力: docs/figures/
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from hanzawa_sim.agents.distill import load_answers
from hanzawa_sim.agents.policy import load_policy_set
from hanzawa_sim.world.loader import load_scenario

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "results"
OUT = ROOT / "docs" / "figures"
DPI = 150

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "gray": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "div": ["#2a78d6", "#f0efec", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "gray": "#4f4e4a",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "seq": ["#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "div": ["#3987e5", "#383835", "#e66767"],
    },
}  # fmt: skip
FOLD = {
    "頭取": "役員（執行役員以上）", "副頭取・専務・常務": "役員（執行役員以上）", "取締役・執行役員": "役員（執行役員以上）",
    "部長": "部長", "次長以下": "次長以下", "出向": "出向", "転籍": "転籍・退職", "退職・退任": "転籍・退職",
    "失脚": "失脚",
}  # fmt: skip
STATES = ["役員（執行役員以上）", "部長", "次長以下", "出向", "転籍・退職", "失脚"]

plt.rcParams.update(
    {
        "font.family": ["Hiragino Sans", "Noto Sans CJK JP", "sans-serif"],
        "font.size": 9.5,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
    }
)


def px(n: float) -> float:
    """ピクセルをポイントに換算する（線の太さ用）。"""
    return n * 72 / DPI


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


SC = load_scenario()
NAME = {p["id"]: p["name"] for p in SC.persons}
SHORT = {
    p["id"]: p["name"][:2] if p["id"] not in ("nakanowatari", "tomari", "ogiso") else p["name"][:3]
    for p in SC.persons
}
SHORT["takagi"] = "高木"


# ---------------------------------------------------------------- 共通の部品
def new_fig(th: dict, w: float, h: float, ncols: int = 1, sharey: bool = False):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), dpi=DPI, sharey=sharey, squeeze=False)
    fig.patch.set_facecolor(th["surface"])
    for ax in axes[0]:
        ax.set_facecolor(th["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(th["axis"])
            ax.spines[side].set_linewidth(px(1))
        ax.tick_params(colors=th["muted"], labelcolor=th["ink2"], width=px(1), length=3)
        ax.title.set_color(th["ink"])
    return fig, axes[0]


def hgrid(ax, th, axis="x"):
    ax.grid(axis=axis, color=th["grid"], linewidth=px(1))
    ax.set_axisbelow(True)


def title(fig, th, text: str, sub: str | None = None):
    fig.text(0.012, 0.975, text, ha="left", va="top", fontsize=11.5, color=th["ink"], weight="bold")
    if sub:
        fig.text(0.012, 0.915, sub, ha="left", va="top", fontsize=8.5, color=th["ink2"])


def rbar(ax, x0: float, y0: float, w: float, h: float, color: str, r_px: float = 4.0, z: float = 2):
    """角を丸めた棒（半径 r_px ピクセル）。軸の範囲と配置を決めた後に呼ぶ。"""
    if w < 0:
        x0, w = x0 + w, -w
    if w <= 0:
        return
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 1))
    ppx, ppy = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
    rx = min(r_px / ppx, w / 2)
    ry = min(r_px / ppy, h / 2)
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={rx}", mutation_aspect=ry / rx if rx > 0 else 1,
            facecolor=color, edgecolor="none", zorder=z,
        )
    )  # fmt: skip


def gap_x(ax, n_px: float) -> float:
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 0))
    return n_px / abs(p1[0] - p0[0])


def inside_ink(th: dict, color: str) -> str:
    yellow = th["series"][3]
    return "#0b0b0b" if color == yellow else "#ffffff"


def save(fig, name: str, mode: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / f"{name}{'' if mode == 'light' else '-dark'}.png", facecolor=fig.get_facecolor(), dpi=DPI
    )
    plt.close(fig)


def legend_row(fig, th, labels: list[str], colors: list[str], y: float, x0: float = 0.012, marker: str = "s"):
    handles = [
        plt.Line2D(
            [], [], marker=marker, linestyle="none", markersize=7, markerfacecolor=c, markeredgecolor="none"
        )
        for c in colors
    ]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x0, y), ncol=len(labels), frameon=False,
               labelcolor=th["ink2"], fontsize=8.5, handletextpad=0.3, columnspacing=1.2)  # fmt: skip


# ---------------------------------------------------------------- 結果の図
BASE = load(RES / "sonnet" / "baseline_100000.json")
N = BASE["n"]


def fig_state_2023(mode: str):
    th = THEMES[mode]
    pids = ["hanzawa", "tomari", "kondo", "fukuyama", "owada", "naito"]
    data = []
    for pid in pids:
        c = Counter()
        for cat, v in BASE["persons"][pid]["final"].items():
            c[FOLD.get(cat, cat)] += v[0]
        data.append([c.get(s, 0.0) for s in STATES])
    fig, (ax,) = new_fig(th, 7.4, 3.9)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.76, bottom=0.12)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(pids) - 0.5, -0.5)
    ax.set_yticks(range(len(pids)), [NAME[p] for p in pids])
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    hgrid(ax, th)
    g = gap_x(ax, 2)
    h = 0.56
    for i, row in enumerate(data):
        left = 0.0
        segs = [(k, v) for k, v in enumerate(row) if v > 0.05]
        for j, (k, v) in enumerate(segs):
            x0 = left + (g / 2 if j > 0 else 0)
            w = v - (g / 2 if j > 0 else 0) - (g / 2 if j < len(segs) - 1 else 0)
            col = th["series"][k]
            rbar(ax, x0, i - h / 2, w, h, col)
            if v >= 7:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color=inside_ink(th, col),
                        zorder=3)  # fmt: skip
            left += v
    title(fig, th, "10年後（2023年9月）の状態", "Sonnet の方策、10万回。数値は%（7%未満は省略）")
    legend_row(fig, th, STATES, th["series"][:6], 0.875, x0=0.005)
    save(fig, "state_2023", mode)


TRAJ = load(RES / "sonnet" / "trajectories_20000.json")


def fig_trajectory(mode: str):
    th = THEMES[mode]
    pids = ["hanzawa", "tomari", "kondo"]
    fig, axes = new_fig(th, 7.6, 3.6, ncols=3, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.74, bottom=0.12, wspace=0.12)
    years = list(range(2014, 2024))
    for ax, pid in zip(axes, pids, strict=True):
        n = TRAJ["n"]
        series = np.zeros((len(STATES), len(years)))
        for j, y in enumerate(years):
            for cat, v in TRAJ["persons"][pid][str(y)].items():
                series[STATES.index(FOLD[cat]), j] += 100 * v / n
        bottom = np.zeros(len(years))
        for k in range(len(STATES)):
            ax.fill_between(years, bottom, bottom + series[k], color=th["series"][k], linewidth=px(2),
                            edgecolor=th["surface"])  # fmt: skip
            bottom += series[k]
        ax.set_xlim(2014, 2023)
        ax.set_ylim(0, 100)
        ax.set_xticks([2014, 2018, 2022])
        ax.set_title(NAME[pid], fontsize=10, color=th["ink"], loc="left")
        ax.spines["left"].set_visible(ax is axes[0])
    axes[0].set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    title(fig, th, "状態の推移（毎年9月末）", "同期3人。Sonnet の方策、2万回")
    legend_row(fig, th, STATES, th["series"][:6], 0.875, x0=0.005)
    save(fig, "trajectory", mode)


def fig_toudori(mode: str):
    th = THEMES[mode]
    years = sorted(BASE["toudori_by_year"])
    names = set(NAME)
    s = {
        "中野渡謙": [],
        "旧T出身（ドラマに登場しない人物）": [],
        "旧S出身（ドラマに登場しない人物）": [],
        "ドラマのその他の人物": [],
    }
    for y in years:
        d = BASE["toudori_by_year"][y]
        s["中野渡謙"].append(d.get("nakanowatari", 0))
        s["旧T出身（ドラマに登場しない人物）"].append(d.get("NPC(旧T)", 0))
        s["旧S出身（ドラマに登場しない人物）"].append(d.get("NPC(旧S)", 0))
        s["ドラマのその他の人物"].append(sum(v for k, v in d.items() if k in names and k != "nakanowatari"))
    fig, (ax,) = new_fig(th, 7.4, 3.6)
    fig.subplots_adjust(left=0.08, right=0.8, top=0.80, bottom=0.12)
    xs = [int(y) for y in years]
    for k, (lab, ys) in enumerate(s.items()):
        col = th["series"][k]
        ax.plot(xs, ys, color=col, linewidth=px(2.5), marker="o", markersize=4, markeredgecolor=th["surface"],
                markeredgewidth=px(2))  # fmt: skip
        short = ["中野渡", "旧T出身", "旧S出身", "ドラマのその他"][k]
        dy = {0: -3.5, 3: 3.5}.get(k, 0.0)  # 0%付近で重なる2本のラベルをずらす
        ax.text(
            xs[-1] + 0.25, ys[-1] + dy, f"{short} {ys[-1]:.1f}%", va="center", fontsize=8.5, color=th["ink2"]
        )
    ax.set_ylim(0, 100)
    ax.set_xlim(xs[0] - 0.3, xs[-1] + 0.2)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    hgrid(ax, th, "y")
    title(fig, th, "頭取の推移（毎年9月末に誰が頭取か）", "Sonnet の方策、10万回")
    legend_row(fig, th, list(s), th["series"][:4], 0.89, x0=0.005, marker="o")
    save(fig, "toudori", mode)


def fig_setbacks(mode: str):
    th = THEMES[mode]
    items = list(BASE["persons"]["hanzawa"]["setback_runs"].items())[:8]
    labels = []
    for k, _ in items:
        lab = k.replace("処分:", "処分: ")
        for pid, nm in NAME.items():
            lab = lab.replace(f"（{pid}）", f"（{nm}）")
        labels.append(lab)
    vals = [v[0] for _, v in items]
    fig, (ax,) = new_fig(th, 7.4, 3.5)
    fig.subplots_adjust(left=0.36, right=0.95, top=0.82, bottom=0.1)
    ax.set_xlim(0, 65)
    ax.set_ylim(len(vals) - 0.5, -0.5)
    ax.set_yticks(range(len(vals)), labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(0, 61, 20), [f"{v}%" for v in range(0, 61, 20)])
    hgrid(ax, th)
    for i, v in enumerate(vals):
        rbar(ax, 0, i - 0.3, v, 0.6, th["series"][0])
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=8.5, color=th["ink2"])
    title(fig, th, "半沢の後退の引き金", "その引き金が10年間に1回以上起きた試行の割合（10万回）")
    save(fig, "hanzawa_setbacks", mode)


def heat(ax, th, M, rows, cols, fmt, cmap, vmin, vmax, text_dark_above):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=th["surface"], linewidth=px(2))
    ax.tick_params(which="minor", length=0)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            if np.isnan(v):
                continue
            light_cell = text_dark_above(v)
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8,
                    color="#0b0b0b" if light_cell else "#ffffff")  # fmt: skip
    return im


def fig_rivalry(mode: str):
    th = THEMES[mode]
    pair = Counter()
    for a, b, _lv, v in BASE["edges"]:
        pair[(a, b)] += 1000 * v / N
    causers = ["kurosaki", "hanzawa", "owada", "tomari"]
    targets = ["hanzawa", "kurosaki", "owada", "fukuyama", "asano", "kaise", "ejima"]
    M = np.array([[pair.get((a, b), 0.0) if a != b else np.nan for b in targets] for a in causers])
    cmap = LinearSegmentedColormap.from_list("seq", th["seq"])
    fig, (ax,) = new_fig(th, 7.2, 3.2)
    fig.subplots_adjust(left=0.12, right=0.9, top=0.78, bottom=0.12)
    vmax = 1100
    im = heat(ax, th, M, [NAME[a] for a in causers], [NAME[b] for b in targets], lambda v: f"{v:.0f}", cmap, 0,
              vmax, (lambda v: v < vmax * 0.45) if mode == "light" else (lambda v: v > vmax * 0.6))  # fmt: skip
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=th["muted"], labelcolor=th["ink2"], labelsize=8)
    title(
        fig,
        th,
        "誰が誰の処分の原因になったか",
        "行: 原因になった人物、列: 処分を受けた人物。1,000試行あたりの件数（降格・出向・解任・刑事告発）",
    )
    save(fig, "rivalry", mode)


ACT = [
    "WORK",
    "LIE_LOW",
    "COVER_UP",
    "MISCONDUCT",
    "RESIGN",
    "GATHER",
    "EXPOSE",
    "BARGAIN",
    "UNDERMINE",
    "LOBBY",
    "SUPPORT",
]
ACT_JA = ["業務専念", "静観", "隠蔽", "不正", "退職", "証拠収集", "告発", "取引", "妨害", "根回し", "支援"]


def fig_actions(mode: str):
    th = THEMES[mode]
    pids = [
        "hanzawa",
        "owada",
        "nakanowatari",
        "tomari",
        "kondo",
        "naito",
        "kurosaki",
        "takagi",
        "fukuyama",
        "asano",
    ]
    M = np.array([[100 * BASE["persons"][p]["actions"].get(a, 0) for a in ACT] for p in pids])
    cmap = LinearSegmentedColormap.from_list("seq", th["seq"])
    fig, (ax,) = new_fig(th, 7.6, 4.4)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.84, bottom=0.08)
    heat(ax, th, M, [NAME[p] for p in pids], ACT_JA, lambda v: f"{v:.0f}", cmap, 0, 65,
         (lambda v: v < 30) if mode == "light" else (lambda v: v > 40))  # fmt: skip
    ax.xaxis.tick_top()
    title(fig, th, "行動の頻度（%）", "10年間に選んだ行動の割合。色が濃いほど多い（Sonnet の方策、10万回）")
    save(fig, "actions", mode)


IV = load(RES / "sonnet" / "interventions_10000.json")
IV_JA = {
    "S2公表_浅野の5000万円": "浅野の5,000万円の件が公になる", "S3公表_黒崎の婚約": "黒崎の婚約の件が公になる",
    "S4公表_大和田と近藤の取引": "大和田と近藤の取引が公になる", "S5公表_半沢の疎開資料": "半沢の疎開資料の件が公になる",
    "中野渡が直後に退任": "中野渡が直後に退任", "大和田が直後に退職": "大和田が直後に退職",
    "半沢が告発しない": "半沢が告発しない", "渡真利が半沢を支援しない": "渡真利が半沢を支援しない",
}  # fmt: skip


def fig_interventions(mode: str):
    th = THEMES[mode]
    keys = list(IV["results"])
    fig, axes = new_fig(th, 7.6, 4.2, ncols=2, sharey=True)
    fig.subplots_adjust(left=0.31, right=0.98, top=0.78, bottom=0.12, wspace=0.08)
    for ax, pid, lab in zip(
        axes, ["hanzawa", "kurosaki"], ["半沢（役員到達率）", "黒崎（金融庁の課長以上）"], strict=True
    ):
        for i, k in enumerate(keys):
            m, h = IV["results"][k]["paired"]["persons"][pid]["ever_exec"]
            sig = abs(m) > h
            col = th["series"][0] if sig else th["gray"]
            ax.plot([m - h, m + h], [i, i], color=col, linewidth=px(2.5), solid_capstyle="round", zorder=2)
            ax.plot([m], [i], marker="o", markersize=6, color=col, markeredgecolor=th["surface"],
                    markeredgewidth=px(2), zorder=3)  # fmt: skip
            if sig and abs(m) >= 5:
                ax.text(m, i - 0.38, f"{m:+.1f}", ha="center", fontsize=8, color=th["ink2"])
        ax.axvline(0, color=th["axis"], linewidth=px(1.5), zorder=1)
        ax.set_ylim(len(keys) - 0.5, -0.8)
        ax.set_title(lab, fontsize=9.5, color=th["ink"], loc="left")
        hgrid(ax, th)
        ax.set_xlabel("変化（ポイント）", color=th["ink2"])
    axes[0].set_xlim(-12, 6)
    axes[1].set_xlim(-40, 20)
    axes[0].set_yticks(range(len(keys)), [IV_JA[k] for k in keys])
    axes[0].tick_params(axis="y", length=0)
    for ax in axes:
        ax.spines["left"].set_visible(False)
    title(
        fig,
        th,
        "介入実験の効果",
        "最初の半期に1つの出来事を起こしたときの変化と95%信頼区間（共通乱数、1万組）",
    )
    legend_row(
        fig,
        th,
        ["0を含まない（効果あり）", "0を含む"],
        [th["series"][0], th["gray"]],
        0.885,
        x0=0.005,
        marker="o",
    )
    save(fig, "interventions", mode)


SENS = load(RES / "sonnet" / "sensitivity_5000.json")
SENS_JA = {
    "inspection_rate": "金融庁検査の頻度", "leak_rate": "偶発的な露見の率", "case_fail_rate": "大型案件の失敗の率",
    "latent_scandal_rate": "派閥の過去の不祥事の率", "board_weight": "後継指名での取締役会の重み",
    "return_base": "出向からの復帰のしやすさ", "age_second_rate": "年次による出向の率",
    "toudori_hazard": "頭取の退任の率", "hr_tau": "人事の選択の温度",
}  # fmt: skip


def fig_sensitivity(mode: str):
    th = THEMES[mode]
    params = list(SENS_JA)
    lo = [SENS["results"][f"{p}:low"]["paired"]["persons"]["hanzawa"]["ever_exec"][0] for p in params]
    hi = [SENS["results"][f"{p}:high"]["paired"]["persons"]["hanzawa"]["ever_exec"][0] for p in params]
    order = sorted(range(len(params)), key=lambda i: -max(abs(lo[i]), abs(hi[i])))
    fig, (ax,) = new_fig(th, 7.4, 3.9)
    fig.subplots_adjust(left=0.3, right=0.96, top=0.78, bottom=0.12)
    ax.set_xlim(-3, 3)
    ax.set_ylim(len(params) - 0.5, -0.5)
    ax.set_yticks(range(len(params)), [SENS_JA[params[i]] for i in order])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    hgrid(ax, th)
    ax.axvline(0, color=th["axis"], linewidth=px(1.5), zorder=1)
    for r, i in enumerate(order):
        rbar(ax, 0, r - 0.34, lo[i], 0.3, th["series"][0])
        rbar(ax, 0, r + 0.04, hi[i], 0.3, th["series"][1])
    ax.set_xlabel("半沢の役員到達率の変化（ポイント）", color=th["ink2"])
    title(
        fig, th, "感度分析: パラメータを変えたときの半沢の役員到達率", "基準の到達率は 24.1%。共通乱数、5千組"
    )
    legend_row(
        fig, th, ["0.5倍（復帰のしやすさは −0.5）", "1.5倍（同 +0.5）"], th["series"][:2], 0.885, x0=0.005
    )
    save(fig, "sensitivity", mode)


HAIKU = load(RES / "haiku" / "baseline_20000.json")


def fig_models(mode: str):
    th = THEMES[mode]
    rows = [
        ("半沢: 役員到達率", lambda b: b["persons"]["hanzawa"]["ever_exec"][0]),
        ("渡真利: 役員到達率", lambda b: b["persons"]["tomari"]["ever_exec"][0]),
        ("近藤: 役員到達率", lambda b: b["persons"]["kondo"]["ever_exec"][0]),
        ("黒崎: 金融庁の課長以上", lambda b: b["persons"]["kurosaki"]["ever_exec"][0]),
        ("半沢: 10年後に失脚", lambda b: b["persons"]["hanzawa"]["final"].get("失脚", [0])[0]),
        ("渡真利: 自主退職", lambda b: b["persons"]["tomari"]["setback_runs"].get("自主退職", [0])[0]),
        ("近藤: 自主退職", lambda b: b["persons"]["kondo"]["setback_runs"].get("自主退職", [0])[0]),
    ]
    fig, (ax,) = new_fig(th, 7.4, 3.6)
    fig.subplots_adjust(left=0.25, right=0.95, top=0.78, bottom=0.12)
    ax.set_xlim(0, 42)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(0, 41, 10), [f"{v}%" for v in range(0, 41, 10)])
    hgrid(ax, th)
    for i, (_, f) in enumerate(rows):
        s, h = f(BASE), f(HAIKU)
        ax.plot([s, h], [i, i], color=th["gray"], linewidth=px(2), zorder=1)
        for v, col in ((s, th["series"][0]), (h, th["series"][1])):
            ax.plot([v], [i], marker="o", markersize=7, color=col, markeredgecolor=th["surface"],
                    markeredgewidth=px(2), zorder=3)  # fmt: skip
    title(
        fig,
        th,
        "LLM のモデルによる違い",
        "同じカードへの回答から学習した方策での結果（Sonnet 10万回、Haiku 2万回）",
    )
    legend_row(fig, th, ["Sonnet", "Haiku"], th["series"][:2], 0.885, x0=0.005, marker="o")
    save(fig, "models", mode)


def fig_backtest(mode: str):
    th = THEMES[mode]
    bt = load(RES / "sonnet" / "backtest.json")
    times = ["2015年4月", "2020年4月"]
    keys = list(bt[times[0]])
    labels = [k.split("（")[0] for k in keys]
    fig, (ax,) = new_fig(th, 7.4, 3.9)
    fig.subplots_adjust(left=0.32, right=0.95, top=0.78, bottom=0.1)
    ax.set_xlim(0, 112)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_yticks(range(len(keys)), labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    hgrid(ax, th)
    for i, k in enumerate(keys):
        for m, t in enumerate(times):
            v = bt[t][k]
            y0 = i - 0.36 + 0.38 * m
            rbar(ax, 0, y0, v, 0.32, th["series"][m])
            ax.text(v + 1.2, y0 + 0.16, f"{v:.0f}%", va="center", fontsize=8, color=th["ink2"])
    title(
        fig,
        th,
        "2020年版との照合",
        "2020年版で描かれた事実が成り立つ試行の割合（Sonnet の方策、8千回）。経過期間の2つの解釈で比べる",
    )
    legend_row(
        fig,
        th,
        ["2015年4月（最終回の1〜2年後と解釈）", "2020年4月（時代設定どおりと解釈）"],
        th["series"][:2],
        0.875,
        x0=0.005,
    )
    save(fig, "backtest", mode)


POL = load(ROOT / "data" / "policies" / "sonnet.json")
HR_JA = ["同じ派閥", "業績", "能力", "自分との関係", "処分歴", "若すぎる", "他の役員の支持"]


def fig_hr(mode: str):
    th = THEMES[mode]
    ds = ["nakanowatari", "takagi", "owada", "naito"]
    M = []
    for d in ds:
        W = np.array(POL["hr"]["person"][d])
        M.append((W[0] - W[1])[1:])
    M = np.array(M)
    cmap = LinearSegmentedColormap.from_list("div", th["div"])
    fig, (ax,) = new_fig(th, 7.2, 2.8)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.72, bottom=0.08)
    im = heat(ax, th, M, [NAME[d] for d in ds], HR_JA, lambda v: f"{v:+.2f}", cmap, -0.8, 0.8,
              (lambda v: abs(v) < 0.6) if mode == "light" else (lambda v: False))  # fmt: skip
    ax.xaxis.tick_top()
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=th["muted"], labelcolor=th["ink2"], labelsize=8)
    title(fig, th, "頭取としての人事の好み", "候補の属性の重み（正: 選ばれやすくなる、負: 選ばれにくくなる）")
    save(fig, "hr_weights", mode)


# ---------------------------------------------------------------- 方法の図
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fig_expose(mode: str):
    th = THEMES[mode]
    e = np.linspace(0, 1, 101)
    cases = [
        ("相手が次長（守り 0.3）", 0.3),
        ("相手が取締役（守り 0.75）", 0.75),
        ("相手が頭取（守り 1.5）", 1.5),
    ]
    cred, sev = 0.4, 0.5
    fig, (ax,) = new_fig(th, 7.4, 3.5)
    fig.subplots_adjust(left=0.09, right=0.64, top=0.8, bottom=0.14)
    for k, (lab, prot) in enumerate(cases):
        p = 100 * sigmoid(4 * (e - 0.5) + 1.5 * sev - 1.5 * prot + cred)
        ax.plot(e, p, color=th["series"][k], linewidth=px(2.5))
        dy = {0: 3.0, 1: -3.0}.get(k, 0.0)
        ax.text(1.02, p[-1] + dy, f"{lab} {p[-1]:.0f}%", va="center", fontsize=8.5, color=th["ink2"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.set_xlabel("告発者が持つ証拠の強さ", color=th["ink2"])
    hgrid(ax, th, "y")
    title(fig, th, "告発の成功確率", "重大度 0.5 の秘密を、部長（信用 0.4）が告発する場合")
    save(fig, "expose_probability", mode)


def fig_sanction(mode: str):
    th = THEMES[mode]
    sev = np.linspace(0, 1, 101)
    fig, (ax,) = new_fig(th, 7.4, 3.6)
    fig.subplots_adjust(left=0.09, right=0.8, top=0.8, bottom=0.14)
    bands = [
        (0, 0.25, "注意"),
        (0.25, 0.45, "降格"),
        (0.45, 0.65, "出向"),
        (0.65, 0.85, "解任"),
        (0.85, 1.05, "刑事告発"),
    ]
    for lo, hi, lab in bands:
        ax.axhline(lo, color=th["grid"], linewidth=px(1), zorder=0)
        ax.text(1.02, (lo + min(hi, 1.0)) / 2, lab, va="center", fontsize=8.5, color=th["ink2"])
    for k, (ev, lab) in enumerate([(1.0, "証拠 1.0"), (0.5, "証拠 0.5")]):
        s = sev * (0.5 + 0.5 * ev)
        ax.fill_between(sev, s - 0.15, s + 0.15, color=th["series"][k], alpha=0.18, linewidth=0)
        ax.plot(sev, s, color=th["series"][k], linewidth=px(2.5))
        ax.text(0.62, 0.62 * (0.5 + 0.5 * ev) + (0.05 if k == 0 else -0.07), lab, fontsize=8.5,
                color=th["ink2"], ha="right" if k == 0 else "left")  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("不正の重大度", color=th["ink2"])
    ax.set_ylabel("処分スコア s", color=th["ink2"])
    title(
        fig,
        th,
        "処分スコアと処分の段階",
        "寛大さ 0 の場合。帯は乱数（±0.15）の範囲。右の文字は s の区間ごとの処分",
    )
    save(fig, "sanction_score", mode)


def fig_distill(mode: str):
    th = THEMES[mode]
    pids = [p["id"] for p in SC.persons]
    pol = load_policy_set(ROOT / "data" / "policies" / "sonnet.json", pids)
    ans = [a for a in load_answers(ROOT / "data" / "elicitation" / "main")
           if a.model == "sonnet" and a.schema == "action" and a.split == "val" and a.variant == "named" and not a.retest]  # fmt: skip
    xs, ys = [], []
    for a in ans:
        dp = pol.named[a.pid]
        z = dp.action_W @ np.concatenate([[1.0], a.x])
        z = np.where(a.mask, z, -np.inf)
        q = np.exp(z - z.max())
        q /= q.sum()
        for k in range(len(a.p)):
            if a.mask[k]:
                xs.append(100 * a.p[k])
                ys.append(100 * q[k])
    fig, (ax,) = new_fig(th, 5.2, 4.6)
    fig.subplots_adjust(left=0.14, right=0.95, top=0.82, bottom=0.12)
    ax.plot([0, 100], [0, 100], color=th["axis"], linewidth=px(1.5), zorder=1)
    ax.scatter(xs, ys, s=10, color=th["series"][0], alpha=0.45, linewidths=0, zorder=2)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.set_xlabel("LLM（Sonnet）が答えた確率 [%]", color=th["ink2"])
    ax.set_ylabel("方策モデルの確率 [%]", color=th["ink2"])
    hgrid(ax, th, "both")
    r = np.corrcoef(xs, ys)[0, 1]
    title(
        fig,
        th,
        "方策モデルと LLM の回答",
        f"検証用カード {len(ans)}枚の各行動（{len(xs)}点）。相関係数 {r:.2f}",
    )
    save(fig, "distill_scatter", mode)
    return r


def fig_validation(mode: str):
    th = THEMES[mode]
    rs = load(ROOT / "data" / "policies" / "report_sonnet.json")["action"]
    full = load(ROOT / "data" / "fullrun" / "sonnet-0" / "summary.json")["on_policy_tv_mean"]
    gap = load(ROOT / "data" / "policies" / "model_gap.json")["tv"]
    items = [
        ("LLM 自体の再回答の差", rs["llm_retest_tv"], False),
        ("方策モデルと LLM（検証用カード）", rs["val_tv_distilled"], True),
        ("方策モデルと LLM（完全LLMラン）", full, True),
        ("人物ごとの平均を返すだけのモデル", rs["val_tv_person_mean"], False),
        ("名前を伏せた版と元の版", rs["anon_vs_named_tv"], False),
        ("Sonnet と Haiku", gap, False),
    ]
    fig, (ax,) = new_fig(th, 7.4, 3.3)
    fig.subplots_adjust(left=0.4, right=0.95, top=0.8, bottom=0.17)
    ax.set_xlim(0, 0.3)
    ax.set_ylim(len(items) - 0.5, -0.5)
    ax.set_yticks(range(len(items)), [i[0] for i in items])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    hgrid(ax, th)
    for i, (_, v, emph) in enumerate(items):
        rbar(ax, 0, i - 0.3, v, 0.6, th["series"][0] if emph else th["gray"])
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8.5, color=th["ink2"])
    ax.set_xlabel("全変動距離（0 = 同じ分布、1 = 重なりなし）", color=th["ink2"])
    title(fig, th, "方策モデルの誤差と比較の基準", "行動の確率分布の差（Sonnet）")
    save(fig, "validation", mode)


def main() -> None:
    figs = [fig_state_2023, fig_trajectory, fig_toudori, fig_setbacks, fig_rivalry, fig_actions, fig_interventions,
            fig_sensitivity, fig_models, fig_backtest, fig_hr, fig_expose, fig_sanction, fig_distill, fig_validation]  # fmt: skip
    for f in figs:
        r = None
        for mode in ("light", "dark"):
            r = f(mode)
        print(f.__name__, "" if r is None else f"r={r:.3f}")


if __name__ == "__main__":
    main()
