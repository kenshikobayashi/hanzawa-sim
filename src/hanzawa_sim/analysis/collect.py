"""多数回の試行を実行しながら、イベントを集計する（全イベントは保存せず、ワーカー内で集計する）。

集計するもの（PLAN.md 5章）:
- 10年後の状態の分布、役員経験、頭取の推移
- 後退（降格・出向・失脚・退職）の直接の引き金と、処分の場合はそれを起こした人物（原因リンクを遡る）
- 昇進の要因分解（人事スコアの項ごとの値。役員人事では次点の候補との差）
- 誰が誰の処分の原因になったか（対立の構造）
- 人物ごとの行動の頻度
- 試行ごとの最終状態（介入実験で、共通乱数による対の比較に使う）
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from hanzawa_sim.agents.policy import PolicySet
from hanzawa_sim.sim.engine import Event, Intervention, Run
from hanzawa_sim.sim.runner import final_category
from hanzawa_sim.world.loader import load_scenario
from hanzawa_sim.world.model import Track
from hanzawa_sim.world.params import Params

SETBACK_TRACKS = {int(Track.SECONDED), int(Track.TRANSFERRED), int(Track.RETIRED), int(Track.OUSTED)}
VIA_JA = {
    "expose": "告発",
    "inspection": "金融庁検査",
    "leak": "偶発的な露見",
    "intervention": "介入",
}
KIND_JA = {
    "case": "大型案件の失敗",
    "latent": "派閥の過去の不祥事",
    "backlash": "告発の失敗",
}
MOVE_JA = {
    "AGE_SECOND": "年次による出向",
    "TRANSFER": "転籍",
    "RETIRE_AGE": "定年・任期",
    "RETIRE_TOUDORI": "頭取の退任",
    "RESIGN": "自主退職",
    "INTERVENTION": "介入",
}


@dataclass
class Agg:
    n: int = 0
    pids: list[str] = field(default_factory=list)
    final: dict = field(default_factory=lambda: defaultdict(Counter))
    max_rank: dict = field(default_factory=lambda: defaultdict(Counter))
    ever_toudori: Counter = field(default_factory=Counter)
    final_toudori: Counter = field(default_factory=Counter)
    toudori_year: Counter = field(default_factory=Counter)  # (年, 頭取)
    setback: dict = field(default_factory=lambda: defaultdict(Counter))  # pid -> 引き金（後退の件数）
    setback_runs: dict = field(
        default_factory=lambda: defaultdict(Counter)
    )  # pid -> 引き金（1回以上起きた試行数）
    sanction_level: dict = field(default_factory=lambda: defaultdict(Counter))
    secret_origin: dict = field(default_factory=lambda: defaultdict(Counter))  # 処分の元になった秘密
    promo_n: Counter = field(default_factory=Counter)
    promo_comp: dict = field(default_factory=lambda: defaultdict(Counter))  # 役員人事: 勝者 − 次点 の和
    promo_n_exec: Counter = field(default_factory=Counter)
    staff_comp: dict = field(default_factory=lambda: defaultdict(Counter))  # 非役員の昇進: 項の値の和
    staff_n: Counter = field(default_factory=Counter)
    edges: Counter = field(default_factory=Counter)  # (原因の人物, 処分を受けた人物, 処分の重さ)
    actions: dict = field(default_factory=lambda: defaultdict(Counter))
    per_run: dict = field(default_factory=dict)  # run_id -> (最終カテゴリの列, 最高ランクの列)

    def merge(self, o: Agg) -> None:
        self.n += o.n
        self.pids = self.pids or o.pids
        for name in ("final", "max_rank", "setback", "setback_runs", "sanction_level", "secret_origin",
                     "promo_comp", "staff_comp", "actions"):  # fmt: skip
            mine = getattr(self, name)
            for k, c in getattr(o, name).items():
                mine[k].update(c)
        for name in (
            "ever_toudori",
            "final_toudori",
            "toudori_year",
            "promo_n",
            "promo_n_exec",
            "staff_n",
            "edges",
        ):
            getattr(self, name).update(getattr(o, name))
        self.per_run.update(o.per_run)


def _label(run: Run, idx: int) -> str:
    if idx < 0:
        return ""
    q = run.w.persons[idx]
    return q.pid if q.named else f"NPC(旧{q.faction})"


def _trace_sanction(run: Run, ev: list[Event], sanction_eid: int) -> tuple[str, str, str]:
    """処分イベントから原因を遡り、(引き金, 原因の人物, 元の秘密) を返す。"""
    se = ev[sanction_eid]
    kind = (se.detail or {}).get("kind", "")
    if kind in KIND_JA:
        actor = ""
        if kind == "backlash":  # 告発に失敗した本人が処分される。告発の相手を原因の人物とする
            ex = ev[se.causes[0]] if se.causes else None
            actor = _label(run, ex.target) if ex else ""
        return f"処分:{KIND_JA[kind]}", actor, ""
    exposed = next((ev[c] for c in se.causes if ev[c].etype == "EXPOSED"), None)
    if exposed is None:
        return "処分:その他", "", ""
    via = VIA_JA.get(exposed.outcome) or exposed.outcome
    s = run.w.secrets[exposed.secret]
    origin = s.sid if s.sid else f"新たな{s.description}"
    actor = _label(run, exposed.actor)
    if kind == "accomplice":
        via += "（共犯）"
    return f"処分:{via}", actor, origin


def compact(run: Run, agg: Agg, cat_index: dict[str, int]) -> None:
    ev = run.events
    named = [q for q in run.w.persons if q.named]
    pids = [q.pid for q in named]
    agg.n += 1
    agg.pids = pids
    summ = run.summary()
    cats = []
    for pid in pids:
        c = final_category(summ[pid])
        agg.final[pid][c] += 1
        agg.max_rank[pid][summ[pid]["max_rank"]] += 1
        cats.append(cat_index.setdefault(c, len(cat_index)))
    agg.per_run[run.run_id] = (
        np.array(cats, dtype=np.int8),
        np.array([summ[p]["max_rank"] for p in pids], np.int8),
    )
    labels = [
        (h if not h.startswith("npc") else _label(run, run.w.person(h).idx)) if h else ""
        for h in run.toudori_hist
    ]
    for pid in set(labels):
        agg.ever_toudori[pid] += 1
    agg.final_toudori[labels[-1]] += 1
    for t in range(1, len(labels), 2):  # 毎年9月末時点
        agg.toudori_year[(2014 + t // 2, labels[t])] += 1

    seen: dict[str, set[str]] = defaultdict(set)
    for e in ev:
        if e.etype == "ACTION":
            agg.actions[_label(run, e.actor)][e.outcome] += 1
            continue
        if e.etype != "MOVE" or e.target < 0 or not run.w.persons[e.target].named:
            continue
        pid = run.w.persons[e.target].pid
        d = e.detail or {}
        fr, to, ft, tt = d["from_rank"], d["to_rank"], d["from_track"], d["to_track"]
        if e.outcome in ("PROMOTION", "TOUDORI") and to > fr:
            agg.promo_n[pid] += 1
            comps = d.get("components") or {}
            if "runner_components" in d:
                agg.promo_n_exec[pid] += 1
                rc = d["runner_components"]
                for k, v in comps.items():
                    agg.promo_comp[pid][k] += v - rc.get(k, 0.0)
            elif comps:
                agg.staff_n[pid] += 1
                for k, v in comps.items():
                    agg.staff_comp[pid][k] += v
            continue
        setback = to < fr or (tt in SETBACK_TRACKS and tt != ft)
        if not setback:
            continue
        if e.outcome == "SANCTION":
            se_id = next((c for c in e.causes if ev[c].etype == "SANCTION"), None)
            if se_id is None:
                continue
            trig, actor, origin = _trace_sanction(run, ev, se_id)
            level = ev[se_id].outcome
            agg.sanction_level[pid][level] += 1
            if origin:
                agg.secret_origin[pid][origin] += 1
            if actor:
                agg.edges[(actor, pid, level)] += 1
            label = f"{trig}（{actor}）" if actor and actor in pids else trig
        else:
            label = MOVE_JA.get(e.outcome, e.outcome)
        agg.setback[pid][label] += 1
        seen[pid].add(label)
    for pid, labs in seen.items():
        for lab in labs:
            agg.setback_runs[pid][lab] += 1


def _worker(args: tuple) -> tuple[Agg, dict[str, int]]:
    seed, run_ids, params, policies, interventions = args
    sc = load_scenario()
    agg = Agg()
    cat_index: dict[str, int] = dict(CAT_INDEX_INIT)
    for rid in run_ids:
        run = Run(sc, params, policies or PolicySet(), seed, rid, interventions)
        run.run(keep_events=True)
        compact(run, agg, cat_index)
    return agg, cat_index


# 最終カテゴリの番号（試行ごとの配列に使う）。ワーカー間で番号をそろえるため、既知のものを先に固定する
CAT_INDEX_INIT = {
    c: i
    for i, c in enumerate(
        [
            "頭取",
            "副頭取・専務・常務",
            "取締役・執行役員",
            "部長",
            "次長以下",
            "出向",
            "転籍",
            "退職・退任",
            "失脚",
            "金融庁:検査官",
            "金融庁:主任検査官",
            "金融庁:統括検査官",
            "金融庁:主任統括検査官",
            "金融庁:課長",
            "金融庁:審議官",
        ]
    )
}


def collect(
    n_runs: int,
    seed: int,
    params: Params | None = None,
    policies: PolicySet | None = None,
    interventions: Sequence[Intervention] = (),
    processes: int | None = None,
    start: int = 0,
) -> tuple[Agg, dict[str, int]]:
    params = params or Params()
    processes = processes or os.cpu_count() or 1
    chunk = max(1, min(250, n_runs // (processes * 4) or 1))
    batches = [
        (seed, list(range(s, min(s + chunk, start + n_runs))), params, policies, tuple(interventions))
        for s in range(start, start + n_runs, chunk)
    ]
    total = Agg()
    cats = dict(CAT_INDEX_INIT)
    with ProcessPoolExecutor(max_workers=processes) as ex:
        for agg, ci in ex.map(_worker, batches):
            unknown = set(ci) - set(cats)
            assert not unknown, f"未登録の最終カテゴリ: {unknown}"
            total.merge(agg)
    return total, cats
