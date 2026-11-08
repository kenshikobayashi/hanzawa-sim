"""完全LLMラン（PLAN.md 4.7 節 (B)）: 名前付きの人物全員の行動を、毎半期 LLM エージェントが決める。

- 各半期、行動する人物全員に、そのときの状況カードと「これまでの出来事」（その人物が知りうるもの）を渡し、
  行動の確率を答えさせる。全員の呼び出しを並列に行う（行動は同時に宣言する規則なので、互いの回答は見えない）
- 回答の確率分布から、試行の乱数表の一様乱数で行動を1つ選ぶ（方策モデルの試行と同じ選び方）
- 取引への応答、頭取としての処分・人事は、学習済みの方策モデルを使う（呼び出し回数を抑えるため）
- 各半期・各人物について、LLM の回答と方策モデルの予測を並べて記録し、試行の中で実際に現れた状態での
  蒸留の誤差を測る
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from hanzawa_sim.agents.cards import Card, describe, render_card, when
from hanzawa_sim.agents.elicit import ACTION_SCHEMA, batch_prompt, normalize_action
from hanzawa_sim.agents.features import build_context
from hanzawa_sim.agents.llm import ClaudeCLI
from hanzawa_sim.agents.policy import DistilledPolicy, PolicySet, sample
from hanzawa_sim.sim.engine import Event, Run
from hanzawa_sim.sim.rng import P
from hanzawa_sim.world.model import BANK_RANK_TITLES, FSA_RANK_TITLES, Action, Org, Person, Track

MOVE_REASON = {
    "PROMOTION": "昇進",
    "TOUDORI": "頭取に就任",
    "RETURN": "出向から銀行に復帰",
    "SANCTION": "処分",
    "AGE_SECOND": "年次による出向",
    "TRANSFER": "出向先に転籍",
    "RETIRE_AGE": "定年・任期で退任",
    "RETIRE_TOUDORI": "頭取を退任",
    "RESIGN": "退職",
}
LEVEL_JA = {
    "warning": "注意",
    "demote": "降格",
    "second": "出向",
    "oust": "解任",
    "arrest": "逮捕",
    "resign": "引責辞任",
}


def _pos(org: int, rank: int, track: int) -> str:
    if org == Org.FSA:
        return f"金融庁 {FSA_RANK_TITLES.get(rank, '')}"
    t = Track(track)
    if t == Track.SECONDED:
        return f"出向（格: {BANK_RANK_TITLES[rank]}）"
    if t in (Track.RETIRED, Track.OUSTED, Track.TRANSFERRED):
        return {Track.RETIRED: "退職", Track.OUSTED: "失脚", Track.TRANSFERRED: "転籍"}[t]
    return BANK_RANK_TITLES[rank]


class LLMRun(Run):
    def __init__(self, *args, backend: ClaudeCLI, prompts: dict[str, str], distilled: PolicySet, **kw):
        super().__init__(*args, **kw)
        self.backend = backend
        self.prompts = prompts  # pid -> システムプロンプト
        self.distilled = distilled
        self.records: list[dict] = []

    # ------------------------------------------------------------ 記憶
    def _event_text(self, e: Event, me: Person) -> str | None:
        w = self.w
        name = lambda i: "あなた" if i == me.idx else describe(self, w.persons[i]).split("（")[0]
        when_txt = when(self.p.start_time + 0.5 * e.step)
        if e.etype == "MOVE":
            q = w.persons[e.target]
            if not q.named and e.target != me.idx:
                return None
            d = e.detail or {}
            reason = MOVE_REASON.get(e.outcome, e.outcome)
            frm = _pos(int(q.org), d["from_rank"], d["from_track"])
            to = _pos(int(q.org), d["to_rank"], d["to_track"])
            return f"{when_txt}: {name(e.target)}が{frm}から{to}へ（{reason}）"
        if e.etype == "EXPOSED":
            via = {"expose": "告発", "inspection": "金融庁検査", "leak": "内部からの露見"}.get(
                e.outcome, e.outcome
            )
            return f"{when_txt}: {name(e.target)}の不正が{via}で公になった"
        if e.etype == "SANCTION" and e.target >= 0 and w.persons[e.target].named:
            return f"{when_txt}: {name(e.target)}が処分を受けた（{LEVEL_JA.get(e.outcome, e.outcome)}）"
        if me.idx not in (e.actor, e.target):
            return None
        if e.etype == "EXPOSE":
            res = "成功" if e.outcome == "success" else "失敗"
            return f"{when_txt}: {name(e.actor)}が{name(e.target)}の不正を告発し、{res}した"
        if e.etype == "BARGAIN":
            r = {"ACCEPT": "応じた", "REJECT": "断った", "RETALIATE": "対抗した"}[e.outcome]
            return f"{when_txt}: {name(e.actor)}が{name(e.target)}に取引を持ちかけ、{name(e.target)}は{r}"
        if e.etype == "ACTION" and e.actor == me.idx:
            return None  # 自分の行動は下の一覧でまとめて示す
        return None

    def memory(self, q: Person, limit: int = 14) -> str:
        items = [t for e in self.events if (t := self._event_text(e, q))]
        mine = [e for e in self.events if e.etype == "ACTION" and e.actor == q.idx]
        lines = items[-limit:]
        if mine:
            recent = "、".join(e.outcome.lower() for e in mine[-4:])
            lines.append(f"あなたが直近の半期に選んだ行動（古い順）: {recent}")
        return "\n".join(f"- {x}" for x in lines) if lines else "- まだ大きな出来事はない"

    # ------------------------------------------------------------ 行動
    def _actions(self) -> None:
        w = self.w
        named = [q for q in w.persons if q.active and q.named]
        ctxs = {q.idx: build_context(self, q) for q in named}
        for q in named:
            for fa in self._forbid.get(q.idx, ()):
                ctxs[q.idx].mask[fa] = False

        def ask(q: Person):
            cid = f"{q.pid}-t{w.step}"
            card = render_card(self, q, ctxs[q.idx], cid)
            text = card + "\n\n## これまでの出来事（あなたが知っていること）\n" + self.memory(q)

            prompt = batch_prompt([Card(cid, q.pid, text, [], [], {})])
            r = self.backend.call(f"full/{self.run_id}/{cid}", self.prompts[q.pid], prompt, ACTION_SCHEMA)
            return q, card, text, r

        with ThreadPoolExecutor(max_workers=len(named) or 1) as ex:
            answers = list(ex.map(ask, named))

        decisions = []
        for q in w.persons:
            if not q.active:
                continue
            u = self.u(P.ACTION, q.idx)
            if q.named:
                ctx = ctxs[q.idx]
                dp = self.distilled.get(q)
                pred = dp.action_probs(ctx) if isinstance(dp, DistilledPolicy) else None
                _, _, text, r = next(a for a in answers if a[0].idx == q.idx)
                dist = None
                if r.ok and r.data and r.data.get("answers"):
                    try:
                        dist = normalize_action(r.data["answers"][0], ctx.mask.tolist())["dist"]
                    except ValueError:
                        dist = None
                if dist is not None:
                    pr = np.array(dist)
                    a = Action(sample(np.log(pr + 1e-12), ctx.mask & (pr > 0), u))
                    src = "llm"
                else:  # 失敗した場合は方策モデルで代える
                    a = dp.choose(q, ctx, u)
                    src = "distilled"
                self.records.append({
                    "step": w.step, "time": w.time, "pid": q.pid, "card": text, "llm": dist,
                    "distilled": None if pred is None else pred.round(4).tolist(), "action": a.name,
                    "source": src, "reason": (r.data or {}).get("answers", [{}])[0].get("reason", "") if r.ok else "",
                })  # fmt: skip
            else:
                ctx = None
                a = self.policies.npc.choose_npc(q, u)
            decisions.append((self.u(P.ORDER, q.idx), q, a, ctx))
        decisions.sort(key=lambda d: d[0])
        for _, q, a, ctx in decisions:
            if q.active:
                self._resolve(q, a, ctx)
