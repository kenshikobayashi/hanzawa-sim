"""世界モデル（ゲームマスター）: 1試行分のシミュレーション。

エージェント（方策）は行動を申請するだけで、成否と状態の更新はすべてここでルールに従って判定する。
半期ごとの処理順序（PLAN.md 4.2 節）:
  1. 外生イベント  2. 行動の選択（同時申請）と判定  3. 人事  4. 関係の時間変化
処分は、露見・告発・案件の失敗などが起きた時点でその場で決める。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hanzawa_sim.agents.features import Context, build_context
from hanzawa_sim.agents.policy import PolicySet
from hanzawa_sim.sim.rng import P, gumbel, make_table, normal
from hanzawa_sim.world.loader import NPC_AGE_RANGE, Scenario, build_world, new_npc
from hanzawa_sim.world.model import (
    ACTIVE_TRACKS,
    BANK_MAX_RANK,
    EXEC_CAPACITY,
    EXEC_MIN_RANK,
    FSA_MAX_RANK,
    IN_BANK,
    MAX_ENTITIES,
    TOUDORI_RANK,
    Action,
    Org,
    Person,
    Response,
    Secret,
    Track,
)
from hanzawa_sim.world.params import Params

SANCTION_LEVELS = ("warning", "demote", "second", "oust", "arrest")


@dataclass(slots=True)
class Event:
    eid: int
    step: int
    etype: str
    actor: int = -1
    target: int = -1
    secret: int = -1
    outcome: str = ""
    value: float = 0.0
    causes: tuple[int, ...] = ()
    detail: dict[str, Any] | None = None


@dataclass
class Intervention:
    """介入: 指定の半期の最初に状態を強制的に変える。

    kind:
      "expose"  target=秘密ID     その秘密を公然にする（通常どおり処分が続く）
      "retire"  target=人物ID     その人物を退職させる
      "trait"   target=人物ID     value=(trait名, 値)
      "forbid"  target=人物ID     value=行動名（Action の名前）。その半期以降、その行動を選べなくする
    """

    step: int
    kind: str
    target: str
    value: Any = None


@dataclass
class RunResult:
    run_id: int
    summary: dict[str, dict[str, Any]]
    toudori: list[str]
    events: list[Event] | None = None


def sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _clip01(v: float) -> float:
    return 0.0 if v < 0.0 else min(v, 1.0)


class Run:
    def __init__(
        self,
        scenario: Scenario,
        params: Params,
        policies: PolicySet,
        seed: int,
        run_id: int,
        interventions: Sequence[Intervention] = (),
        debug: bool = False,
    ):
        self.p = params
        self.run_id = run_id
        self.U = make_table(seed, run_id, params.n_steps)
        self.w = build_world(scenario, params, self.U[0, P.INIT])
        self.policies = policies
        self.interventions = list(interventions)
        self.debug = debug
        self.events: list[Event] = []
        self.toudori_hist: list[str] = []
        self._u = self.U[0]
        self._sanction_slot: dict[int, int] = {}
        self._forbid: dict[int, set[Action]] = {}
        self._n_moves = 0
        self._board_key: tuple[int, int] = (-1, -1)
        self._board_cache: tuple[np.ndarray, np.ndarray] = (np.zeros(0, dtype=np.int64), np.zeros(0))
        self._inactive: set[int] = {q.idx for q in self.w.persons if not q.active}
        self.max_rank = {q.idx: q.rank for q in self.w.persons if q.named}
        self.counts: dict[int, dict[str, int]] = {q.idx: {} for q in self.w.persons if q.named}

    # ------------------------------------------------------------------ 共通
    def u(self, purpose: P, entity: int, slot: int = 0) -> float:
        return float(self._u[purpose, entity, slot])

    def z(self, purpose: P, entity: int, slot: int = 0) -> float:
        return normal(self.u(purpose, entity, slot), self.u(purpose, entity, slot + 1))

    def log(self, etype: str, **kw: Any) -> int:
        eid = len(self.events)
        self.events.append(Event(eid, self.w.step, etype, **kw))
        return eid

    def _add(self, mat: np.ndarray, i: int, j: int, v: float) -> None:
        mat[i, j] = _clip01(float(mat[i, j]) + v)

    def _board(self) -> tuple[np.ndarray, np.ndarray]:
        """取締役以上（銀行内・ランク6以上）の人物の添字と影響力。人事異動か人物の追加があれば作り直す。"""
        key = (len(self.w.persons), self._n_moves)
        if self._board_key != key:
            b = [k for k in self.w.persons if k.in_bank and k.rank >= 6]
            self._board_cache = (
                np.array([k.idx for k in b], dtype=np.int64),
                np.array([k.rank / 10.0 for k in b]),
            )
            self._board_key = key
        return self._board_cache

    def sponsor(self, c: Person, exclude: int = -1) -> float:
        """後ろ盾の強さ: 取締役以上の人物のうち、c を最も強く支持する人物の 影響力×支持。"""
        w = self.w
        idx, power = self._board()
        keep = (idx != c.idx) & (idx != exclude)
        if not keep.any():
            return 0.0
        b = idx[keep]
        v = w.trust[b, c.idx] + 0.5 * w.debt[b, c.idx] - w.hostility[b, c.idx]
        v = np.clip(v.astype(np.float64), 0.0, 1.0) * power[keep]
        return max(0.0, float(v.max()))

    def _fresh(self, s: Secret) -> float:
        return 0.5 ** ((self.w.time - s.created) / self.p.secret_halflife)

    def _secrets_of(self, i: int, public: bool = False) -> list[Secret]:
        return [s for s in self.w.secrets if s.public == public and (s.subject == i or i in s.accomplices)]

    def _count(self, q: Person, key: str) -> None:
        if q.named:
            c = self.counts[q.idx]
            c[key] = c.get(key, 0) + 1

    # ------------------------------------------------------------------ 実行
    def run(self, keep_events: bool = True) -> RunResult:
        for t in range(self.p.n_steps):
            self.step(t)
        return RunResult(self.run_id, self.summary(), self.toudori_hist, self.events if keep_events else None)

    def step(self, t: int) -> None:
        w = self.w
        w.step = t
        self._u = self.U[t]
        self._sanction_slot.clear()
        for q in w.persons:
            q.attacked_last, q.attacked = q.attacked, False
        exec_step = t % 2 == 1  # 4〜9月の半期（6月の株主総会）に役員人事
        self._apply_interventions(t)
        self._exogenous()
        self._actions()
        self._hr(exec_step)
        self._decay_relations()
        if self.debug:
            self.check_invariants(exec_step)
        td = w.toudori()
        self.toudori_hist.append(td.pid if td else "")
        w.time += 0.5

    # ------------------------------------------------------------------ 状態の変更
    def move(
        self,
        q: Person,
        rank: int,
        track: Track,
        etype: str,
        causes: Iterable[int] = (),
        actor: int = -1,
        detail: dict[str, Any] | None = None,
    ) -> int:
        """ランク・所属区分の変更。必ずこの関数を通し、イベントとして記録する。"""
        w = self.w
        if self.debug and etype == "PROMOTION" and q.org == Org.BANK and EXEC_MIN_RANK <= rank < 10:
            assert w.occupancy(rank) < EXEC_CAPACITY[rank], f"満席のランク{rank}への昇進"
        was_toudori = q.in_bank and q.rank == TOUDORI_RANK
        d = {"from_rank": q.rank, "to_rank": rank, "from_track": int(q.track), "to_track": int(track)}
        if detail:
            d.update(detail)
        old_rank = q.rank
        self._n_moves += 1
        if rank != old_rank:
            q.rank_since = w.time
        q.rank, q.track = rank, track
        eid = self.log(
            "MOVE",
            actor=actor,
            target=q.idx,
            outcome=etype,
            value=float(rank),
            causes=tuple(causes),
            detail=d,
        )
        if q.named:
            self.max_rank[q.idx] = max(self.max_rank[q.idx], rank)
            if rank > old_rank:
                self._count(q, "promotions")
            elif rank < old_rank:
                self._count(q, "demotions")
            if track == Track.SECONDED and d["from_track"] != int(Track.SECONDED):
                self._count(q, "secondments")
        if not q.active:
            self._inactive.add(q.idx)
        if was_toudori and not (q.in_bank and q.rank == TOUDORI_RANK):
            self._succession(q, forced=(etype == "SANCTION"), cause=eid)
        return eid

    def _apply_interventions(self, t: int) -> None:
        w = self.w
        for iv in self.interventions:
            if iv.step != t:
                continue
            eid = self.log("INTERVENTION", outcome=iv.kind, detail={"target": iv.target, "value": iv.value})
            if iv.kind == "expose":
                s = next(s for s in w.secrets if s.sid == iv.target)
                self._make_public(s, (eid,), "intervention")
            elif iv.kind == "retire":
                q = w.person(iv.target)
                if q.active:
                    self.move(q, q.rank, Track.RETIRED, "INTERVENTION", causes=(eid,))
            elif iv.kind == "trait":
                name, value = iv.value
                w.person(iv.target).traits[name] = float(value)
            elif iv.kind == "forbid":
                self._forbid.setdefault(w.person(iv.target).idx, set()).add(Action[iv.value])
            else:
                raise ValueError(f"未知の介入: {iv.kind}")

    # ------------------------------------------------------------------ 外生イベント
    def _exogenous(self) -> None:
        w, p = self.w, self.p
        for q in w.persons:
            if not q.active:
                continue
            q.perf = (
                p.perf_persist * q.perf
                + (1 - p.perf_persist) * 2 * (q.ability - 0.5)
                + p.perf_noise * self.z(P.PERF, q.idx, 0)
            )
            q.perf = max(-3.0, min(3.0, q.perf))

        # 大型案件の失敗
        for q in list(w.persons):
            if not q.active or q.org != Org.BANK or q.rank < 2:
                continue
            prob = p.case_fail_rate * (1.5 if q.perf < -0.5 else 1.0) * (1.3 - 0.6 * q.ability)
            if self.u(P.CASE, q.idx, 0) < prob:
                sev = 0.15 + 0.35 * self.u(P.CASE, q.idx, 1)
                eid = self.log("CASE_FAILURE", target=q.idx, value=sev)
                q.perf -= 0.8
                self._sanction(q.idx, sev, 1.0, (eid,), "case")

        # 金融庁検査
        if self.u(P.GLOBAL, 0, 0) < p.inspection_rate:
            self._inspection()

        # 偶発的な露見（知っている人物が多く、証拠が強く、新しい秘密ほど起きやすい）
        for s in list(w.secrets):
            if s.public or not w.persons[s.subject].active:
                continue
            ev = [
                e
                for k, e in s.knowers.items()
                if k != s.subject and k not in s.accomplices and w.persons[k].active
            ]
            if not ev:
                continue
            prob = p.leak_rate * len(ev) * max(ev) * self._fresh(s)
            if self.u(P.DISCOVERY, s.idx, 0) < prob:
                eid = self.log("LEAK", target=s.subject, secret=s.idx)
                self._make_public(s, (eid,), "leak")

        # 派閥の未知の過去の不祥事（2020年版の具体的な不祥事は使わず、一般化して扱う）
        if self.u(P.GLOBAL, 1, 0) < p.latent_scandal_rate:
            fac = "S" if self.u(P.GLOBAL, 1, 1) < 0.5 else "T"
            execs = [q for q in w.persons if q.is_exec and q.faction == fac]
            if execs:
                resp = max(execs, key=lambda q: (q.rank, -q.birth))
                sev = 0.4 + 0.5 * self.u(P.GLOBAL, 1, 2)
                eid = self.log("LATENT_SCANDAL", target=resp.idx, outcome=fac, value=sev)
                self._sanction(resp.idx, sev, 1.0, (eid,), "latent")

    def _inspector(self) -> Person | None:
        for q in self.w.persons:
            if q.active and q.org == Org.FSA and q.rank <= self.p.fsa_inspector_max_rank:
                return q
        return None

    def _inspection(self) -> None:
        w, p = self.w, self.p
        kuro = self._inspector()
        eid = self.log("INSPECTION", actor=kuro.idx if kuro else -1)
        found = 0
        for s in list(w.secrets):
            subj = w.persons[s.subject]
            if s.public or not subj.active or subj.org != Org.BANK:
                continue
            prob = (p.insp_find + (p.insp_kurosaki_bonus if kuro else 0.0)) * self._fresh(s)
            if kuro:
                prob += 0.25 * s.evidence(kuro.idx)
            if self.u(P.DISCOVERY, s.idx, 1) < prob:
                self._make_public(s, (eid,), "inspection", exposer=kuro.idx if kuro else -1)
                found += 1
        if kuro:
            kuro.perf += 0.3 if found else -0.15

    # ------------------------------------------------------------------ 露見と処分
    def _make_public(self, s: Secret, causes: tuple[int, ...], via: str, exposer: int = -1) -> None:
        if s.public:
            return
        s.public = True
        eid = self.log("EXPOSED", actor=exposer, target=s.subject, secret=s.idx, outcome=via, causes=causes)
        if s.sanctioned:
            return
        s.sanctioned = True
        mult = 1.2 if via == "inspection" else 1.0
        ev = max(s.knowers.values()) if s.knowers else 1.0
        self._sanction(s.subject, min(1.0, s.severity * mult), ev, (eid,), "misconduct", secret=s.idx)
        for a in s.accomplices:
            self._sanction(a, 0.7 * s.severity * mult, ev, (eid,), "accomplice", secret=s.idx)

    def _sanction(
        self, ti: int, severity: float, evidence: float, causes: tuple[int, ...], kind: str, secret: int = -1
    ) -> int:
        w, p = self.w, self.p
        t = w.persons[ti]
        if not t.active:
            return -1
        slot = min(self._sanction_slot.get(ti, 0), 7)
        self._sanction_slot[ti] = slot + 1
        s = severity * (0.5 + 0.5 * evidence) + (self.u(P.SANCTION, ti, slot) - 0.5) * 2 * p.sanction_noise
        td = w.toudori()
        decider = None
        leniency = 0.0
        if t.org == Org.BANK and td is not None and td.idx != ti:
            decider = td
            leniency = p.leniency_scale * td.trait("org_protect") * (1.0 if t.rank >= 6 else 0.3)
            leniency += (
                p.crony_leniency * td.trait("cronyism") * float(w.trust[td.idx, ti] - w.hostility[td.idx, ti])
            )
        s -= leniency
        is_toudori = t.in_bank and t.rank == TOUDORI_RANK
        W = self.policies.sanction.get(decider.pid) if decider is not None and decider.named else None
        probs = None
        if is_toudori:
            level = "resign" if s >= p.toudori_resign_threshold else "warning"
        elif W is not None and severity * (0.5 + 0.5 * evidence) < p.sanction_thresholds[-1]:
            # 頭取が名前付きの人物なら、LLMの回答から学習した処分の判断を使う（刑事事件になる重さは除く）
            d = decider.idx
            rel = float(np.clip((w.trust[d, ti] - w.hostility[d, ti]) / 0.4, -1.0, 1.0))
            x = np.array(
                [1.0, severity, evidence, float(t.rank >= 5), float(t.faction == decider.faction), rel]
            )
            z = W @ x
            probs = np.exp(z - z.max())
            probs /= probs.sum()
            k = int(np.searchsorted(np.cumsum(probs), self.u(P.SANCTION, ti, slot) * probs.sum()))
            level = SANCTION_LEVELS[min(k, 3)]
            leniency = float(probs[0])  # 記録用: 注意のみで済ませる確率
        else:
            level = SANCTION_LEVELS[sum(s >= th for th in p.sanction_thresholds)]
        eid = self.log(
            "SANCTION",
            actor=decider.idx if decider else -1,
            target=ti,
            secret=secret,
            outcome=level,
            value=s,
            causes=causes,
            detail={
                "kind": kind,
                "severity": severity,
                "evidence": evidence,
                "leniency": leniency,
                "distilled": probs is not None,
            },
        )
        t.scandal += (0.2 if level == "warning" else 0.5) * severity
        t.reputation = max(-2.0, t.reputation - (0.2 if level == "warning" else 0.5))
        if level == "warning":
            return eid
        if level == "resign":
            self.move(t, t.rank, Track.RETIRED, "SANCTION", causes=(eid,))
        elif level == "demote":
            self.move(
                t,
                max(1, t.rank - 1),
                t.track,
                "SANCTION",
                causes=(eid,),
                actor=decider.idx if decider else -1,
            )
        elif level == "second":
            if t.org == Org.FSA:  # 金融庁では他部局・他省庁への異動（降格扱い）
                self.move(t, max(1, t.rank - 1), t.track, "SANCTION", causes=(eid,))
                t.perf = 0.0
            else:
                new_track = Track.TRANSFERRED if t.track == Track.SECONDED else Track.SECONDED
                self.move(
                    t, t.rank, new_track, "SANCTION", causes=(eid,), actor=decider.idx if decider else -1
                )
        else:  # oust / arrest
            self.move(
                t, t.rank, Track.OUSTED, "SANCTION", causes=(eid,), detail={"arrest": level == "arrest"}
            )
        return eid

    # ------------------------------------------------------------------ 行動
    def _actions(self) -> None:
        w = self.w
        decisions: list[tuple[float, Person, Action, Context | None]] = []
        for q in w.persons:
            if not q.active:
                continue
            u = self.u(P.ACTION, q.idx)
            if q.named:
                ctx = build_context(self, q)
                for fa in self._forbid.get(q.idx, ()):
                    ctx.mask[fa] = False
                a = self.policies.get(q).choose(q, ctx, u)
            else:
                ctx = None
                a = self.policies.npc.choose_npc(q, u)
            decisions.append((self.u(P.ORDER, q.idx), q, a, ctx))
        decisions.sort(key=lambda d: d[0])
        for _, q, a, ctx in decisions:
            if q.active:
                self._resolve(q, a, ctx)

    def _resolve(self, q: Person, a: Action, ctx: Context | None) -> None:
        w, p = self.w, self.p
        i = q.idx
        eid = -1
        if q.named:
            eid = self.log("ACTION", actor=i, outcome=a.name)
        if a == Action.WORK:
            q.perf += p.work_gain * q.ability
            q.reputation = min(2.0, q.reputation + 0.03)
        elif a == Action.LIE_LOW:
            w.hostility[:, i] *= 0.9
        elif a == Action.COVER_UP:
            for s in self._secrets_of(i):
                for k in list(s.knowers):
                    if k != i and self.u(P.OUTCOME, i, 0) < 0.7:
                        s.knowers[k] *= 0.6
                if self.u(P.OUTCOME, i, 1) < 0.1:
                    s.severity = min(1.0, s.severity + 0.1)
        elif a == Action.MISCONDUCT:
            q.perf += p.misconduct_gain
            s = self._new_secret(i, 0.3 + 0.4 * self.u(P.OUTCOME, i, 0), {i: 1.0}, "不正")
            watchers = [k for k in w.persons if k.active and k.idx != i and w.hostility[k.idx, i] > 0.3]
            if watchers and self.u(P.OUTCOME, i, 1) < 0.3:
                k = max(watchers, key=lambda k: w.hostility[k.idx, i])
                s.knowers[k.idx] = 0.3
        elif a == Action.RESIGN:
            self.move(q, q.rank, Track.RETIRED, "RESIGN", causes=(eid,), actor=i)
        elif ctx is None:
            if a == Action.LOBBY:
                self._lobby(q, self._npc_patron(q))
        elif a == Action.GATHER:
            self._gather(q, ctx.rival, eid)
        elif a == Action.EXPOSE:
            self._expose(q, ctx.expose_secret, (eid,), slot0=0)
        elif a == Action.BARGAIN:
            self._bargain(q, ctx, eid)
        elif a == Action.UNDERMINE:
            self._undermine(q, ctx.rival, eid, slot0=0)
        elif a == Action.LOBBY:
            self._lobby(q, ctx.patron)
        elif a == Action.SUPPORT:
            self._support(q, ctx.ally, eid)

    def _new_secret(self, subject: int, severity: float, knowers: dict[int, float], desc: str) -> Secret:
        idx = len(self.w.secrets)
        if idx >= MAX_ENTITIES:
            raise RuntimeError(f"秘密の数が上限 {MAX_ENTITIES} を超えた")
        s = Secret(idx, f"X{idx}", subject, (), severity, knowers, self.w.time, description=desc)
        self.w.secrets.append(s)
        return s

    def _npc_patron(self, q: Person) -> int:
        w = self.w
        td = w.toudori()
        if td is not None and td.faction == q.faction and td.idx != q.idx:
            return td.idx
        execs = [k for k in w.persons if k.is_exec and k.faction == q.faction and k.rank > q.rank]
        return max(execs, key=lambda k: k.rank).idx if execs else -1

    def _gather(self, q: Person, ti: int, eid: int) -> None:
        w, p = self.w, self.p
        if ti < 0 or not w.persons[ti].active:
            return
        i = q.idx
        info = q.trait("info_skill", 1.0)
        targets = self._secrets_of(ti)
        known = [s for s in targets if i in s.knowers]
        if known:
            s = max(known, key=lambda s: s.severity)
            s.knowers[i] = _clip01(s.knowers[i] + (0.1 + 0.2 * self.u(P.OUTCOME, i, 0)) * info)
            s.sources.setdefault(i, []).append(eid)
        else:
            unknown = [s for s in targets if i not in s.knowers]
            if unknown:
                s = max(unknown, key=lambda s: s.severity)
                if self.u(P.OUTCOME, i, 0) < p.gather_discover * info * self._fresh(s):
                    s.knowers[i] = 0.3
                    s.sources.setdefault(i, []).append(eid)
                    self.log("DISCOVER", actor=i, target=ti, secret=s.idx, causes=(eid,))
        if self.u(P.OUTCOME, i, 1) < p.gather_detect:
            self._add(w.hostility, ti, i, 0.2)
            w.persons[ti].attacked = True

    def _expose(self, q: Person, si: int, causes: tuple[int, ...], slot0: int) -> None:
        w, p = self.w, self.p
        if si < 0:
            return
        s = w.secrets[si]
        t = w.persons[s.subject]
        if s.public or not t.active:
            return
        i = q.idx
        e = s.evidence(i)
        td = w.toudori()
        protection = w.power(t) + 0.5 * self.sponsor(t) + (0.5 if td is not None and td.idx == t.idx else 0.0)
        cred = 0.5 * q.reputation + w.power(q)
        logit = (
            p.expose_k_evidence * (e - 0.5)
            + p.expose_k_severity * s.severity
            - p.expose_k_protection * protection
            + p.expose_k_credibility * cred
            + (p.expose_fsa_bonus if q.org == Org.FSA else 0.0)
        )
        prob = sigmoid(logit)
        ok = self.u(P.OUTCOME, i, slot0) < prob
        eid = self.log(
            "EXPOSE",
            actor=i,
            target=t.idx,
            secret=si,
            outcome="success" if ok else "fail",
            value=prob,
            causes=tuple(causes) + tuple(s.sources.get(i, [])),
        )
        t.attacked = True
        if ok:
            self._add(w.hostility, t.idx, i, 0.4)
            q.reputation = min(2.0, q.reputation + 0.2)
            self._make_public(s, (eid,), "expose", exposer=i)
        else:
            self._add(w.hostility, t.idx, i, 0.5)
            if self.u(P.OUTCOME, i, slot0 + 1) < 0.5:
                self._sanction(i, 0.25 + 0.4 * w.power(t), 1.0, (eid,), "backlash")

    def _bargain(self, q: Person, ctx: Context, eid: int) -> None:
        w = self.w
        i = q.idx
        if ctx.lever_secret >= 0:
            s = w.secrets[ctx.lever_secret]
            t = w.persons[s.subject]
            kind, lev = "threat", s.evidence(i) * s.severity
            ev, sev = s.evidence(i), s.severity
        elif ctx.coopt_target >= 0:
            t = w.persons[ctx.coopt_target]
            kind, lev = "offer", w.power(q)
            held = [s2 for s2 in self._secrets_of(i) if t.idx in s2.knowers]
            s2 = max(held, key=lambda s2: s2.evidence(t.idx)) if held else None
            ev, sev = (s2.evidence(t.idx), s2.severity) if s2 else (0.0, 0.0)
        else:
            return
        if not t.active or t.idx == i:
            return
        tctx = build_context(self, t) if t.named else None
        # 応答の特徴（special_cards.RESPONSE_FEATURES と同じ順）
        rx = np.array(
            [
                float(kind == "threat"),
                ev,
                sev,
                w.power(q) - w.power(t),
                float(w.trust[t.idx, i]),
                float(w.hostility[t.idx, i]),
                t.rank / (6.0 if t.org == Org.FSA else 10.0),
            ]
        )
        resp = self.policies.get(t).respond(t, tctx, kind, lev, self.u(P.RESPONSE, t.idx), rx=rx)
        beid = self.log(
            "BARGAIN",
            actor=i,
            target=t.idx,
            outcome=resp.name,
            value=lev,
            causes=(eid,),
            detail={"kind": kind},
        )
        if resp == Response.ACCEPT:
            if kind == "threat":
                self._add(w.debt, t.idx, i, 0.6 * lev)
                w.hostility[i, t.idx] *= 0.8
            else:  # 懐柔: 相手は自分の秘密の証拠を手放す（近藤が報告書を渡さなかった取引に相当）
                self._add(w.trust, t.idx, i, 0.3)
                self._add(w.debt, t.idx, i, 0.3)
                w.hostility[t.idx, i] *= 0.5
                for s2 in self._secrets_of(i):
                    if t.idx in s2.knowers:
                        s2.knowers[t.idx] *= 0.3
        elif resp == Response.REJECT:
            self._add(w.hostility, t.idx, i, 0.15)
        else:  # 報復
            self._add(w.hostility, t.idx, i, 0.4)
            q.attacked = True
            mine = [s2 for s2 in self._secrets_of(i) if s2.evidence(t.idx) >= 0.3]
            if mine:
                self._expose(
                    t, max(mine, key=lambda s2: s2.evidence(t.idx) * s2.severity).idx, (beid,), slot0=4
                )
            else:
                self._undermine(t, i, beid, slot0=4)

    def _undermine(self, q: Person, ti: int, eid: int, slot0: int) -> None:
        w, p = self.w, self.p
        if ti < 0 or not w.persons[ti].active:
            return
        i = q.idx
        t = w.persons[ti]
        ok = self.u(P.OUTCOME, i, slot0) < p.undermine_success
        if ok:
            t.perf -= 0.4
            t.attacked = True
        if q.named or t.named:
            self.log(
                "UNDERMINE",
                actor=i,
                target=ti,
                outcome="success" if ok else "fail",
                causes=(eid,) if eid >= 0 else (),
            )
        if self.u(P.OUTCOME, i, slot0 + 1) < p.undermine_detect:
            self._new_secret(i, 0.3, {i: 1.0, ti: 0.6}, "妨害")
            self._add(w.hostility, ti, i, 0.3)

    def _lobby(self, q: Person, ki: int) -> None:
        w, p = self.w, self.p
        if ki < 0:
            return
        i = q.idx
        gain = p.lobby_gain * (1 - float(w.trust[ki, i])) * (1 - float(w.hostility[ki, i]))
        self._add(w.trust, ki, i, gain)

    def _support(self, q: Person, ai: int, eid: int) -> None:
        """味方の業績を助け、味方の敵についての情報を共有する（渡真利の役割に相当）。"""
        w, p = self.w, self.p
        if ai < 0 or not w.persons[ai].active:
            return
        i = q.idx
        ally = w.persons[ai]
        ally.perf += p.support_gain
        self._add(w.trust, ai, i, 0.1)
        for s in w.secrets:
            e = s.knowers.get(i, 0.0)
            if s.public or e <= 0 or s.subject == ai or w.hostility[ai, s.subject] <= 0.4:
                continue
            if s.knowers.get(ai, 0.0) < 0.8 * e:
                s.knowers[ai] = 0.8 * e
                s.sources.setdefault(ai, []).append(eid)
                self.log("SHARE", actor=i, target=ai, secret=s.idx, causes=(eid,))

    # ------------------------------------------------------------------ 人事
    def _hr_components(
        self, decider: Person | None, c: Person, target_rank: int, board_weight: float
    ) -> dict[str, float]:
        """人事スコアの項ごとの値。decider=None は取締役会のみによる判断。"""
        w, p, W = self.w, self.p, self.p.hr_w
        beta = self.policies.hr.get(decider.pid) if decider is not None and decider.named else None
        if beta is not None:
            return self._hr_components_distilled(decider, beta, c, target_rank, board_weight)
        rel = fac = 0.0
        dw = 0.0 if decider is None else 1.0 - board_weight
        if decider is not None:
            d = decider.idx
            rel += (
                dw
                * decider.trait("cronyism")
                * float(w.trust[d, c.idx] - w.hostility[d, c.idx] + 0.5 * w.debt[d, c.idx])
            )
            if decider.faction in ("S", "T"):
                fac += dw * decider.trait("faction_bias") * (1.0 if c.faction == decider.faction else -1.0)
        bw = 1.0 - dw
        if bw > 0:
            board = [b for b in w.persons if b.is_exec and b.rank >= 6 and b.idx != c.idx]
            if board:
                rel += (
                    bw
                    * 0.5
                    * float(
                        np.mean(
                            [
                                w.trust[b.idx, c.idx] - w.hostility[b.idx, c.idx] + 0.5 * w.debt[b.idx, c.idx]
                                for b in board
                            ]
                        )
                    )
                )
                same = sum(b.faction == c.faction for b in board) / len(board)
                fac += bw * 0.5 * (2 * same - 1)
        return {
            "perf": W["perf"] * c.perf,
            "ability": W["ability"] * (c.ability - 0.5) * 2,
            "sponsor": W["sponsor"] * self.sponsor(c, exclude=decider.idx if decider else -1),
            "relation": W["relation"] * rel,
            "faction": W["faction"] * fac,
            "age": -W["age"] * max(0.0, c.age(w.time) - (p.exec_age_limit[target_rank] - 4))
            - W["young"] * max(0.0, p.rank_min_age[target_rank] - c.age(w.time)),
            "scandal": -W["scandal"] * c.scandal,
            "reputation": W["reputation"] * c.reputation,
        }

    def _hr_components_distilled(
        self, d: Person, beta: np.ndarray, c: Person, target_rank: int, board_weight: float
    ) -> dict[str, float]:
        """頭取が名前付きの人物のとき: 頭取の好み（LLMの回答から学習した重み）＋取締役会の支持。

        候補の属性は special_cards.HR_FEATURES の順に、カードの水準（0/1 または −1/0/1）へ換算する。
        選択は score / hr_tau + Gumbel なので、2択の回答のロジット尺度に合わせて hr_tau を掛ける。
        """
        w, p = self.w, self.p
        board = self._hr_components(None, c, target_rank, 1.0)
        comps = {k: board_weight * v for k, v in board.items() if k != "age"} if board_weight > 0 else {}
        comps["age"] = board["age"]
        rel = w.trust[d.idx, c.idx] - w.hostility[d.idx, c.idx] + 0.5 * w.debt[d.idx, c.idx]
        xc = np.array(
            [
                float(c.faction == d.faction),
                float(np.clip(c.perf, -1.0, 1.0)),
                float(np.clip((c.ability - 0.5) / 0.3, 0.0, 1.0)),
                float(np.clip(rel / 0.4, -1.0, 1.0)),
                min(1.0, c.scandal / 0.2),
                float(c.age(w.time) < p.rank_min_age[target_rank]),
                float(np.clip(self.sponsor(c, exclude=d.idx) / 0.5, 0.0, 1.0)),
            ]
        )
        comps["decider"] = (1.0 - board_weight) * p.hr_tau * float(beta @ xc)
        return comps

    def _select(
        self, decider: Person | None, cands: list[Person], target_rank: int, board_weight: float = 0.0
    ) -> tuple[Person, dict[str, Any]]:
        scored = []
        for c in cands:
            comps = self._hr_components(decider, c, target_rank, board_weight)
            s = sum(comps.values())
            g = gumbel(self.u(P.HR, c.idx, target_rank - 3))
            scored.append((s / self.p.hr_tau + g, s, c, comps))
        scored.sort(key=lambda x: -x[0])
        _, s, winner, comps = scored[0]
        detail: dict[str, Any] = {"components": comps, "score": s, "n_candidates": len(cands)}
        if len(scored) > 1:
            detail["runner_up"] = scored[1][2].pid
            detail["runner_components"] = scored[1][3]
        detail["decider"] = decider.pid if decider else "board"
        return winner, detail

    def _succession(self, outgoing: Person, forced: bool, cause: int) -> None:
        """頭取の後継。通常は前頭取の指名（取締役会の支持も加味）、不祥事による辞任では取締役会が選ぶ。"""
        w, p = self.w, self.p
        lim = p.exec_age_limit[10] - 2
        cands: list[Person] = []
        for lo, hi in ((7, 9), (6, 6), (5, 5)):
            cands = [q for q in w.persons if q.in_bank and lo <= q.rank <= hi and q.age(w.time) < lim]
            if cands:
                break
        if not cands:
            raise RuntimeError("頭取の候補者がいない")
        decider = None if forced else outgoing
        chosen, detail = self._select(decider, cands, 10, board_weight=p.board_weight)
        detail["forced"] = forced
        self.move(
            chosen, TOUDORI_RANK, Track.HQ, "TOUDORI", causes=(cause,), actor=outgoing.idx, detail=detail
        )

    def _hr(self, exec_step: bool) -> None:
        w, p = self.w, self.p
        if exec_step:
            td = w.toudori()
            if td is not None:
                age = td.age(w.time)
                hz = p.toudori_hazard + (p.toudori_old_hazard if age >= p.toudori_old_age else 0.0)
                if age >= p.exec_age_limit[10] or self.u(P.GLOBAL, 2, 0) < hz:
                    self.move(td, td.rank, Track.RETIRED, "RETIRE_TOUDORI")
            for q in list(w.persons):
                if q.is_exec and q.rank < 10 and q.age(w.time) >= p.exec_age_limit[q.rank]:
                    self.move(q, q.rank, Track.RETIRED, "RETIRE_AGE")
            self._fill_exec()
            self._fsa_hr()
        self._staff_hr()
        self._replenish_npcs()

    def _fill_exec(self) -> None:
        w, p = self.w, self.p
        td = w.toudori()
        for r in range(9, EXEC_MIN_RANK - 1, -1):
            while w.occupancy(r) < EXEC_CAPACITY[r]:
                lim = p.exec_age_limit[r] - 0.5
                cands = [
                    q
                    for q in w.persons
                    if q.in_bank
                    and q.rank == r - 1
                    and q.age(w.time) < lim
                    and w.time - q.rank_since >= p.min_tenure_exec
                ]
                if not cands:
                    break
                chosen, detail = self._select(td, cands, r)
                self.move(chosen, r, Track.HQ, "PROMOTION", actor=td.idx if td else -1, detail=detail)

    def _staff_hr(self) -> None:
        """非役員の昇進、出向からの復帰、年次による出向・転籍、定年退職。"""
        w, p = self.w, self.p
        td = w.toudori()
        for q in list(w.persons):
            if not q.active or q.org != Org.BANK:
                continue
            age = q.age(w.time)
            td_rel = (
                float(w.trust[td.idx, q.idx] - w.hostility[td.idx, q.idx]) if td and td.idx != q.idx else 0.0
            )
            if q.in_bank and q.rank <= 3 and q.named and w.time - q.rank_since >= p.min_tenure_staff:
                comps = {
                    "base": p.promo_base[q.rank],
                    "perf": q.perf,
                    "ability": 1.2 * (q.ability - 0.5) * 2,
                    "sponsor": self.sponsor(q),
                    "relation": 0.8 * td_rel,
                    "scandal": -1.5 * q.scandal,
                    "reputation": 0.3 * q.reputation,
                    "age": -0.5 * max(0.0, age - p.promo_age_max[q.rank])
                    - p.hr_w["young"] * max(0.0, p.rank_min_age[q.rank + 1] - age),
                }
                prob = sigmoid(sum(comps.values()))
                if self.u(P.PROMO, q.idx, 0) < prob:
                    self.move(q, q.rank + 1, q.track, "PROMOTION", detail={"components": comps, "prob": prob})
            elif q.track == Track.SECONDED and q.rank <= 4 and age < p.return_max_age:
                comps = {
                    "base": p.return_base,
                    "perf": q.perf,
                    "ability": 0.5 * (q.ability - 0.5) * 2,
                    "sponsor": self.sponsor(q),
                    "relation": td_rel,
                    "scandal": -1.5 * q.scandal,
                }
                prob = sigmoid(sum(comps.values()))
                if self.u(P.PROMO, q.idx, 1) < prob:
                    self.move(q, q.rank, Track.HQ, "RETURN", detail={"components": comps, "prob": prob})
            if q.in_bank and q.rank <= 4 and age >= p.age_second_threshold[q.rank]:
                if self.u(P.PROMO, q.idx, 2) < p.age_second_rate * (1 - 0.6 * self.sponsor(q)):
                    self.move(q, q.rank, Track.SECONDED, "AGE_SECOND")
            elif q.track == Track.SECONDED and age >= p.transfer_age:
                if self.u(P.PROMO, q.idx, 3) < p.transfer_rate:
                    self.move(q, q.rank, Track.TRANSFERRED, "TRANSFER")
            if (
                q.active
                and (q.track == Track.SECONDED or (q.in_bank and q.rank <= 4))
                and age >= p.staff_retire_age
            ):
                self.move(q, q.rank, Track.RETIRED, "RETIRE_AGE")

    def _fsa_hr(self) -> None:
        w, p = self.w, self.p
        for q in list(w.persons):
            if not q.active or q.org != Org.FSA:
                continue
            if q.age(w.time) >= p.fsa_retire_age:
                self.move(q, q.rank, Track.RETIRED, "RETIRE_AGE")
                continue
            prob = sigmoid(p.fsa_promo_base + q.perf - 1.5 * q.scandal)
            tenure_ok = w.time - q.rank_since >= p.min_tenure_fsa
            if q.rank < FSA_MAX_RANK and tenure_ok and self.u(P.PROMO, q.idx, 4) < prob:
                self.move(q, q.rank + 1, q.track, "PROMOTION", detail={"prob": prob})

    def _replenish_npcs(self) -> None:
        w, p = self.w, self.p
        n = sum(1 for q in w.persons if not q.named and q.in_bank and q.rank == 4)
        for k in range(max(0, p.npc_pool_size - n)):
            u = self._u[P.NPC, k]
            new_npc(w, 4, Track.HQ if u[0] < 0.5 else Track.BRANCH, u, NPC_AGE_RANGE[4])
            q = w.persons[-1]
            w.pid_index[q.pid] = q.idx

    def _decay_relations(self) -> None:
        w, p = self.w, self.p
        n = len(w.persons)
        g = np.array([q.trait("grudge") for q in w.persons], dtype=np.float32)
        keep = p.hostility_decay_min + (1 - p.hostility_decay_min) * g
        w.hostility[:n, :n] *= keep[:, None]
        w.trust[:n, :n] *= p.trust_decay
        w.debt[:n, :n] *= p.debt_decay

    # ------------------------------------------------------------------ 検査と集計
    def check_invariants(self, exec_step: bool) -> None:
        w, p = self.w, self.p
        tds = [q for q in w.persons if q.in_bank and q.rank == TOUDORI_RANK]
        assert len(tds) == 1, f"頭取が{len(tds)}人"
        for q in w.persons:
            maxr = FSA_MAX_RANK if q.org == Org.FSA else BANK_MAX_RANK
            assert 1 <= q.rank <= maxr, f"{q.pid} のランク {q.rank} が範囲外"
            if q.idx in self._inactive:
                assert not q.active, f"{q.pid} が非活動から復帰した"
            if q.in_bank and q.rank <= 4:
                assert q.age(w.time) < p.staff_retire_age, f"{q.pid} が定年を超えて在籍"
            if exec_step and q.is_exec and q.rank < 10:
                assert q.age(w.time) < p.exec_age_limit[q.rank], f"{q.pid} が役員の退任年齢を超えて在任"
        for s in w.secrets:
            assert all(0.0 <= e <= 1.0 for e in s.knowers.values()), f"{s.sid} の証拠が範囲外"
        n = len(w.persons)
        for m in (w.trust, w.hostility, w.debt):
            assert float(m[:n, :n].min()) >= 0.0 and float(m[:n, :n].max()) <= 1.0

    def summary(self) -> dict[str, dict[str, Any]]:
        w = self.w
        out = {}
        for q in w.persons:
            if not q.named:
                continue
            out[q.pid] = {
                "rank": q.rank,
                "track": int(q.track),
                "org": int(q.org),
                "max_rank": self.max_rank[q.idx],
                "ever_toudori": q.pid in self.toudori_hist,
                **self.counts[q.idx],
            }
        return out


def run_one(
    scenario: Scenario,
    params: Params,
    seed: int,
    run_id: int,
    policies: PolicySet | None = None,
    interventions: Sequence[Intervention] = (),
    keep_events: bool = False,
    debug: bool = False,
) -> RunResult:
    run = Run(scenario, params, policies or PolicySet(), seed, run_id, interventions, debug)
    return run.run(keep_events=keep_events)


__all__ = ["ACTIVE_TRACKS", "IN_BANK", "Event", "Intervention", "Run", "RunResult", "run_one"]
