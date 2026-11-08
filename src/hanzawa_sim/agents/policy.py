"""行動方策。

- HeuristicPolicy: 人物の traits から行動のロジットを計算する暫定のルール（動作確認用。分析には使わない）
- DistilledPolicy: LLM エージェントの回答から学習した多項ロジット（agents/distill.py）
- PolicySet: 人物ごとの方策の対応表。頭取としての処分の判断・人事の好みのモデルも持つ
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from hanzawa_sim.agents.features import Context, F
from hanzawa_sim.world.model import N_ACTIONS, Action, Person, Response


def sample(logits: np.ndarray, mask: np.ndarray, u: float) -> int:
    """マスク付きソフトマックスから、一様乱数 u で1つ選ぶ（逆関数法）。"""
    z = np.where(mask, logits, -np.inf)
    z = z - z.max()
    pr = np.exp(z)
    c = np.cumsum(pr)
    k = int(np.searchsorted(c, u * c[-1]))
    return min(k, len(logits) - 1)


class Policy(Protocol):
    def choose(self, q: Person, ctx: Context, u: float) -> Action: ...

    def respond(
        self,
        q: Person,
        ctx: Context | None,
        kind: str,
        leverage: float,
        u: float,
        rx: np.ndarray | None = None,
    ) -> Response: ...


class HeuristicPolicy:
    """traits に基づく暫定の行動ルール。"""

    def logits(self, q: Person, ctx: Context) -> np.ndarray:
        x = ctx.x
        amb, agg = q.trait("ambition"), q.trait("aggression")
        integ, loy = q.trait("integrity"), q.trait("loyalty")
        ev, exp_, att = x[F["evidence_on_rival"]], x[F["my_exposure"]], x[F["attacked"]]
        L = np.zeros(N_ACTIONS)
        L[Action.WORK] = 1.0 + 0.5 * amb
        L[Action.LIE_LOW] = 0.2 + (1 - agg) * exp_ + 0.5 * att * (1 - agg)
        L[Action.COVER_UP] = -0.5 + 2.5 * (1 - integ) * exp_
        L[Action.MISCONDUCT] = -3.0 + 2.5 * (1 - integ) * amb
        L[Action.RESIGN] = -4.0 + 2.0 * x[F["seconded"]] * float(x[F["years_left"]] < 0.3) + att * (1 - amb)
        L[Action.GATHER] = -0.5 + 2.0 * agg * (1 - ev)
        L[Action.EXPOSE] = -1.5 + 3.5 * agg * ev + att
        L[Action.BARGAIN] = -1.0 + 2.0 * (1 - integ) * ev + amb * float(x[F["rank"]] >= 0.6)
        L[Action.UNDERMINE] = -2.0 + 2.0 * agg * (1 - integ) + 0.5 * att
        L[Action.LOBBY] = -0.5 + 1.5 * amb * (1 - x[F["sponsor"]])
        L[Action.SUPPORT] = -0.8 + 1.5 * loy
        return L

    def choose(self, q: Person, ctx: Context, u: float) -> Action:
        return Action(sample(self.logits(q, ctx), ctx.mask, u))

    def respond(
        self,
        q: Person,
        ctx: Context | None,
        kind: str,
        leverage: float,
        u: float,
        rx: np.ndarray | None = None,
    ) -> Response:
        amb, agg = q.trait("ambition"), q.trait("aggression")
        integ, loy = q.trait("integrity"), q.trait("loyalty")
        if kind == "threat":
            L = np.array([1.5 + 2.5 * leverage - 1.5 * agg, 0.5 * integ, -0.5 + 2.0 * agg])
        else:
            L = np.array([-1.0 + 1.5 * amb + (1 - loy) + leverage, 0.8 * integ, -1.5 + agg])
        return Response(sample(L, np.ones(3, dtype=bool), u))


class NpcPolicy:
    """NPC の行動: ほとんど業務専念。根回しと不正を少し。"""

    def choose_npc(self, q: Person, u: float) -> Action:
        p_mis = 0.06 * (1 - q.trait("integrity"))
        p_lobby = 0.12 * q.trait("ambition") / 0.55
        p_low = 0.05
        if u < p_mis:
            return Action.MISCONDUCT
        if u < p_mis + p_lobby:
            return Action.LOBBY
        if u < p_mis + p_lobby + p_low:
            return Action.LIE_LOW
        return Action.WORK

    def respond(
        self,
        q: Person,
        ctx: Context | None,
        kind: str,
        leverage: float,
        u: float,
        rx: np.ndarray | None = None,
    ) -> Response:
        p_acc = 1 / (1 + math.exp(-(2.0 * leverage - 0.5)))
        if u < p_acc:
            return Response.ACCEPT
        return Response.REJECT if u < p_acc + 0.8 * (1 - p_acc) else Response.RETALIATE


class DistilledPolicy:
    """学習した多項ロジット。action_W: (行動数, 1+特徴数)、response_W: (3, 1+応答の特徴数)。"""

    def __init__(self, action_W: np.ndarray, response_W: np.ndarray):
        self.action_W = action_W
        self.response_W = response_W

    def action_probs(self, ctx: Context) -> np.ndarray:
        z = self.action_W @ np.concatenate([[1.0], ctx.x])
        z = np.where(ctx.mask, z, -np.inf)
        e = np.exp(z - z.max())
        return e / e.sum()

    def choose(self, q: Person, ctx: Context, u: float) -> Action:
        return Action(sample(self.action_W @ np.concatenate([[1.0], ctx.x]), ctx.mask, u))

    def respond(
        self,
        q: Person,
        ctx: Context | None,
        kind: str,
        leverage: float,
        u: float,
        rx: np.ndarray | None = None,
    ) -> Response:
        assert rx is not None, "学習済みの応答モデルには応答の特徴 rx が必要"
        return Response(sample(self.response_W @ np.concatenate([[1.0], rx]), np.ones(3, dtype=bool), u))


class PolicySet:
    """人物ごとの方策の対応表。指定のない名前付き人物は HeuristicPolicy、NPC は NpcPolicy。

    sanction[pid]: 頭取としての処分の判断の多項ロジット (4, 1+5)。
    hr[pid]: 頭取としての人事の好み（候補の属性の重み、長さ7）。
    どちらも、その人物が頭取のときだけ使い、ない場合はエンジンの規則で決める。
    """

    def __init__(
        self,
        named: dict[str, Policy] | None = None,
        sanction: dict[str, np.ndarray] | None = None,
        hr: dict[str, np.ndarray] | None = None,
        name: str = "heuristic",
    ):
        self.named = named or {}
        self.sanction = sanction or {}
        self.hr = hr or {}
        self.name = name
        self.default = HeuristicPolicy()
        self.npc = NpcPolicy()

    def get(self, q: Person):
        if not q.named:
            return self.npc
        return self.named.get(q.pid, self.default)


def load_policy_set(path: Path, pids: list[str]) -> PolicySet:
    """scripts/fit_policies.py が保存した JSON から PolicySet を作る。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))

    def W(block: dict, pid: str) -> np.ndarray:
        return np.array(block["person"].get(pid, block["shared"]))

    named: dict[str, Policy] = {
        pid: DistilledPolicy(W(d["action"], pid), W(d["response"], pid)) for pid in pids
    }
    sanction = {pid: np.array(m) for pid, m in d["sanction"]["person"].items()}
    hr = {}
    for pid, m in d["hr"]["person"].items():
        m = np.array(m)
        hr[pid] = (m[0] - m[1])[1:]  # 候補A − 候補B の重み（切片＝提示順の偏りは除く）
    return PolicySet(named, sanction, hr, name=Path(path).stem)
