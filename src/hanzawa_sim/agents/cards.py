"""状況カード: その人物から見た状態を文章にしたもの。

カードの内容は agents/features.py の Context（モンテカルロで方策が受け取る特徴）と1対1に対応させる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hanzawa_sim.agents.features import FEATURES, Context, F, build_context
from hanzawa_sim.agents.persona import ACTION_KEYS
from hanzawa_sim.agents.policy import PolicySet
from hanzawa_sim.sim.engine import Run
from hanzawa_sim.world.loader import Scenario
from hanzawa_sim.world.model import BANK_RANK_TITLES, FSA_RANK_TITLES, Action, Org, Person, Secret, Track
from hanzawa_sim.world.params import Params


@dataclass
class Card:
    card_id: str
    pid: str
    text: str
    x: list[float]
    mask: list[bool]
    meta: dict = field(default_factory=dict)


def when(time: float) -> str:
    year = int(time)
    return f"{year}年{10 if time - year >= 0.5 else 4}月"


def describe(run: Run, q: Person) -> str:
    """人物の呼び方と立場。NPC は役職と出身行で示す。"""
    if q.org == Org.FSA:
        pos = f"金融庁 {FSA_RANK_TITLES[q.rank]}"
    elif q.track == Track.SECONDED:
        pos = f"出向中（出向前の格: {BANK_RANK_TITLES[q.rank]}）"
    else:
        pos = BANK_RANK_TITLES[q.rank]
    fac = {"S": "旧S", "T": "旧T"}.get(q.faction, "")
    if not q.named:  # ドラマに登場しない人物は役職と出身行だけで示す
        return f"{fac}出身の{pos}"
    return f"{q.name}（{pos}{'・' + fac + '出身' if fac else ''}）"


# シミュレーション中に生まれる秘密（engine の _new_secret）の説明を文章にする
DYNAMIC_SECRET_TEXT = {
    "不正": "規則に反する手段で目先の成果を上げた（審査の規則に反する融資の処理など）",
    "妨害": "敵対する相手の仕事を妨害し、失敗の責任を負わせた",
}


# 秘密の説明の中の、読み手本人の姓を「あなた」に置き換えるための姓
SURNAMES = {
    "hanzawa": "半沢", "owada": "大和田", "nakanowatari": "中野渡", "tomari": "渡真利", "kondo": "近藤",
    "naito": "内藤", "kurosaki": "黒崎", "takagi": "高木", "kishikawa": "岸川", "fukuyama": "福山",
    "asano": "浅野", "ogiso": "小木曽", "ejima": "江島", "kaise": "貝瀬", "furusato": "古里",
}  # fmt: skip


def secret_text(s: Secret, viewer: Person | None = None) -> str:
    text = DYNAMIC_SECRET_TEXT.get(s.description, s.description)
    if viewer is not None and viewer.pid in SURNAMES:
        text = text.replace(SURNAMES[viewer.pid], "あなた")
    return text


def _level(v: float, hi: float, lo: float, words: tuple[str, str, str]) -> str:
    return words[0] if v >= hi else words[1] if v >= lo else words[2]


def render_card(run: Run, q: Person, ctx: Context, card_id: str) -> str:
    w, x = run.w, ctx.x
    age = q.age(w.time)
    lines = [f"【状況 {card_id}】{when(w.time)}時点"]
    if q.org == Org.FSA:
        lines.append(f"- あなたの立場: 金融庁 {FSA_RANK_TITLES[q.rank]}（{age:.0f}歳）")
    elif q.track == Track.SECONDED:
        lines.append(
            f"- あなたの立場: 東京中央銀行から出向中。出向前の格は{BANK_RANK_TITLES[q.rank]}（{age:.0f}歳）"
        )
    else:
        unit = "本部" if q.track == Track.HQ else "支店"
        lines.append(f"- あなたの立場: 東京中央銀行 {BANK_RANK_TITLES[q.rank]}（{unit}、{age:.0f}歳）")
    lines.append(f"- 定年・任期の上限までの残り: 約{x[F['years_left']] * 10:.0f}年")
    lines.append(f"- 直近の業績: {_level(x[F['perf']], 0.25, -0.25, ('好調', '普通', '不振'))}")

    td = w.toudori()
    if x[F["is_toudori"]]:
        lines.append("- 頭取: あなた自身")
    elif td is not None:
        attitude = _level(x[F["toudori_relation"]], 0.3, -0.1, ("好意的", "中立", "冷淡"))
        who = td.name if td.named else f"旧{td.faction}出身の人物（ドラマに登場しない）"
        lines.append(f"- 頭取: {who}。あなたへの態度は{attitude}")
        if q.org == Org.BANK:
            lines.append(f"- 頭取の派閥: {'あなたと同じ' if x[F['faction_in_power']] else 'あなたとは別'}")

    if ctx.rival >= 0:
        r = w.persons[ctx.rival]
        diff = x[F["rival_power_diff"]]
        power = _level(diff, 0.15, -0.15, ("あなたより大きい", "あなたと同程度", "あなたより小さい"))
        lines.append(f"- 最も敵対している相手: {describe(run, r)}。影響力は{power}")
        if ctx.expose_secret >= 0:
            s = w.secrets[ctx.expose_secret]
            e = s.evidence(q.idx)
            strength = _level(e, 0.8, 0.4, ("確かな証拠", "ある程度の証拠", "疑いを持つ程度の弱い証拠"))
            lines.append(f"- 相手の弱み: 「{secret_text(s, q)}」について、あなたは{strength}を持っている")
        else:
            lines.append("- 相手の弱み: 今のところ握っていない")
    else:
        lines.append("- 強く敵対している相手: いない")

    exposure = x[F["my_exposure"]]
    if exposure > 0:
        lines.append(
            f"- あなたの弱み: あなたの秘密を他に知っている者がいる（危険度: {_level(exposure, 0.5, 0.25, ('高い', '中程度', '低い'))}）"
        )
    else:
        lines.append("- あなたの弱み: 他人に握られている秘密はない")

    sp = x[F["sponsor"]]
    lines.append(f"- 役員の後ろ盾: {_level(sp, 0.5, 0.2, ('強い', '弱い', 'ほぼない'))}")
    if ctx.ally >= 0:
        lines.append(f"- 頼れる味方: {describe(run, w.persons[ctx.ally])}")
    if ctx.patron >= 0:
        lines.append(f"- 近づける有力者: {describe(run, w.persons[ctx.patron])}")
    lines.append(f"- 前の半期に攻撃を受けた: {'はい' if x[F['attacked']] else 'いいえ'}")
    if q.org == Org.BANK and q.track != Track.SECONDED:
        lines.append(f"- 1つ上の役職の空席: {'ある' if x[F['vacancy_above']] else 'ない'}")
    ok = [ACTION_KEYS[a] for a in Action if ctx.mask[a]]
    ng = [ACTION_KEYS[a] for a in Action if not ctx.mask[a]]
    lines.append(f"- 選べる行動: {', '.join(ok)}")
    if ng:
        lines.append(f"- 選べない行動: {', '.join(ng)}")
    return "\n".join(lines)


def sample_cards(
    sc: Scenario,
    pids: list[str],
    n_per_person: int,
    seed: int,
    n_runs: int = 20,
    params: Params | None = None,
    frac_far: float = 1.0,
) -> dict[str, list[Card]]:
    """暫定ルールのシミュレーションで実際に現れた状態からカードを作る。

    人物ごとに、全スナップショットの特徴ベクトルから互いに最も離れたものを順に選ぶ（最遠点選択）。
    最遠点選択は珍しい状態に偏るため、frac_far < 1 のときは残りを無作為に選び、よく現れる状態も含める。
    """
    params = params or Params()
    pool: dict[str, list[tuple[np.ndarray, Card]]] = {pid: [] for pid in pids}
    for rid in range(n_runs):
        run = Run(sc, params, PolicySet(), seed, rid)
        for t in range(params.n_steps):
            run.step(t)
            for pid in pids:
                q = run.w.person(pid)
                if not q.active:
                    continue
                ctx = build_context(run, q)
                ctx.x[F["attacked"]] = float(q.attacked)  # 直前の半期（step t）の攻撃
                cid = f"{pid}-r{rid}s{t}"
                card = Card(
                    cid,
                    pid,
                    render_card(run, q, ctx, cid),
                    ctx.x.tolist(),
                    ctx.mask.tolist(),
                    {"run": rid, "step": t, "time": run.w.time},
                )
                pool[pid].append((ctx.x.astype(float), card))
    out: dict[str, list[Card]] = {}
    for pid, items in pool.items():
        if not items:
            out[pid] = []
            continue
        X = np.array([v for v, _ in items])
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        n = min(n_per_person, len(items))
        n_far = max(1, round(n * frac_far))
        chosen = [0]
        d = np.linalg.norm(X - X[0], axis=1)
        while len(chosen) < n_far:
            k = int(np.argmax(d))
            chosen.append(k)
            d = np.minimum(d, np.linalg.norm(X - X[k], axis=1))
        rest = np.setdiff1d(np.arange(len(items)), chosen)
        rng = np.random.default_rng([seed, len(pid)])
        chosen += rng.choice(rest, size=n - len(chosen), replace=False).tolist()
        out[pid] = [items[k][1] for k in chosen]
    return out


__all__ = ["FEATURES", "Card", "describe", "render_card", "sample_cards", "secret_text", "when"]
