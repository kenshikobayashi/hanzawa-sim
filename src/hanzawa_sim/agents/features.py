"""人物から見た状態の特徴（状況カードの生成とモンテカルロで共通に使う）。

PLAN.md 4.7 節: カードの生成とモンテカルロで同じ特徴抽出関数を使い、両者の状態の定義を一致させる。
各人物はその人物が知っている秘密だけを特徴に反映する（部分観測）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from hanzawa_sim.world.model import (
    EXEC_CAPACITY,
    N_ACTIONS,
    Action,
    Org,
    Person,
    Track,
)

if TYPE_CHECKING:
    from hanzawa_sim.sim.engine import Run

FEATURES = (
    "rank",  # 自分のランク/10（金融庁はランク/6）
    "seconded",  # 出向中
    "external",  # 銀行外の組織に所属
    "years_left",  # 定年・任期上限までの年数/10（0〜1）
    "perf",  # 直近の業績（−1〜1）
    "has_rival",  # 敵がいる
    "rival_power_diff",  # 敵の影響力 − 自分の影響力
    "evidence_on_rival",  # 敵の秘密について自分が持つ証拠の強さ
    "my_exposure",  # 自分の非公然の秘密を他人が知っている度合い（×重大度）
    "sponsor",  # 後ろ盾の強さ
    "toudori_relation",  # 頭取の自分への態度（信頼−敵意）
    "faction_in_power",  # 頭取が自分と同じ派閥
    "attacked",  # 前の半期に攻撃を受けた
    "vacancy_above",  # 1つ上のランクに空席がある
    "is_toudori",  # 自分が頭取
    "has_ally",  # 味方がいる
)
N_FEATURES = len(FEATURES)
F = {name: i for i, name in enumerate(FEATURES)}


@dataclass(slots=True)
class Context:
    x: np.ndarray
    mask: np.ndarray
    rival: int = -1
    ally: int = -1
    patron: int = -1
    expose_secret: int = -1  # 告発に使う秘密（敵の非公然の秘密のうち 証拠×重大度 が最大）
    lever_secret: int = -1  # 取引の材料にできる秘密（証拠0.3以上）
    coopt_target: int = -1  # 懐柔の相手（自分の秘密の証拠を握っている人物）
    known: dict = field(default_factory=dict)


def years_left(run: Run, q: Person) -> float:
    p, age = run.p, q.age(run.w.time)
    if q.org == Org.FSA:
        limit = p.fsa_retire_age
    elif q.in_bank and q.rank >= 5:
        limit = p.exec_age_limit[q.rank]
    else:
        limit = p.staff_retire_age
    return max(0.0, min(10.0, limit - age)) / 10.0


def build_context(run: Run, q: Person) -> Context:
    w = run.w
    i = q.idx
    H, T = w.hostility, w.trust
    others = [k for k in w.persons if k.active and k.idx != i]

    rival = max(others, key=lambda k: H[i, k.idx], default=None)
    rival_i = rival.idx if rival is not None and H[i, rival.idx] >= 0.2 else -1
    allies = [k for k in others if k.idx != rival_i]
    ally = max(allies, key=lambda k: T[i, k.idx], default=None)
    ally_i = ally.idx if ally is not None and T[i, ally.idx] >= 0.3 else -1

    my_power = w.power(q)
    patron_i, best = -1, 0.0
    for k in others:
        if not k.in_bank or w.power(k) <= my_power:
            continue
        v = w.power(k) * (0.5 + T[k.idx, i] - H[k.idx, i])
        if v > best:
            patron_i, best = k.idx, v

    # 敵の秘密と自分の秘密
    ev_rival, expose_i, best_score = 0.0, -1, 0.0
    exposure = 0.0
    coopt_i, coopt_ev = -1, 0.0
    for s in w.secrets:
        if s.public:
            continue
        if rival_i >= 0 and (s.subject == rival_i or rival_i in s.accomplices):
            e = s.knowers.get(i, 0.0)
            if e > 0:
                ev_rival = max(ev_rival, e)
                if e * s.severity > best_score:
                    expose_i, best_score = s.idx, e * s.severity
        if s.subject == i or i in s.accomplices:
            holders = [(k, e) for k, e in s.knowers.items() if k != i and w.persons[k].active]
            if holders:
                exposure = max(exposure, s.severity * max(e for _, e in holders))
                if my_power >= 0.6:
                    for k, e in holders:
                        if e >= 0.3 and e > coopt_ev and w.persons[k].named:
                            coopt_i, coopt_ev = k, e
    lever_i = expose_i if expose_i >= 0 and w.secrets[expose_i].knowers.get(i, 0.0) >= 0.3 else -1

    td = w.toudori()
    if td is None:
        td_rel = 0.0
    elif td.idx == i:
        td_rel = 1.0
    else:
        td_rel = float(np.clip(T[td.idx, i] - H[td.idx, i], -1.0, 1.0))
    vacancy = 0.0
    if q.in_bank and q.rank + 1 in EXEC_CAPACITY and q.rank + 1 < 10:
        vacancy = float(w.occupancy(q.rank + 1) < EXEC_CAPACITY[q.rank + 1])

    x = np.zeros(N_FEATURES, dtype=np.float32)
    x[F["rank"]] = q.rank / (6.0 if q.org == Org.FSA else 10.0)
    x[F["seconded"]] = float(q.track == Track.SECONDED)
    x[F["external"]] = float(q.org != Org.BANK)
    x[F["years_left"]] = years_left(run, q)
    x[F["perf"]] = float(np.clip(q.perf / 2.0, -1.0, 1.0))
    x[F["has_rival"]] = float(rival_i >= 0)
    x[F["rival_power_diff"]] = (w.power(w.persons[rival_i]) - my_power) if rival_i >= 0 else 0.0
    x[F["evidence_on_rival"]] = ev_rival
    x[F["my_exposure"]] = exposure
    x[F["sponsor"]] = run.sponsor(q)
    x[F["toudori_relation"]] = td_rel
    x[F["faction_in_power"]] = float(td is not None and td.faction == q.faction)
    x[F["attacked"]] = float(q.attacked_last)
    x[F["vacancy_above"]] = vacancy
    x[F["is_toudori"]] = float(td is not None and td.idx == i)
    x[F["has_ally"]] = float(ally_i >= 0)

    mask = np.ones(N_ACTIONS, dtype=bool)
    mask[Action.COVER_UP] = exposure > 0
    mask[Action.RESIGN] = q.org == Org.BANK
    mask[Action.GATHER] = rival_i >= 0
    mask[Action.UNDERMINE] = rival_i >= 0
    mask[Action.EXPOSE] = expose_i >= 0
    mask[Action.BARGAIN] = lever_i >= 0 or coopt_i >= 0
    mask[Action.LOBBY] = patron_i >= 0
    mask[Action.SUPPORT] = ally_i >= 0
    return Context(
        x=x,
        mask=mask,
        rival=rival_i,
        ally=ally_i,
        patron=patron_i,
        expose_secret=expose_i,
        lever_secret=lever_i,
        coopt_target=coopt_i,
    )
