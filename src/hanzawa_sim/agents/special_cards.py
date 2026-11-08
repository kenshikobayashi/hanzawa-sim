"""行動カード以外のカード。

- 取引への応答（R）: 取引を持ちかけられた人物の応答。暫定ルールのシミュレーションで取引が起こりうる状態から作る
- 処分の判断（J）: 頭取として不正を処分する重さ。要因を組み合わせた架空の状況
- 人事の好み（H）: 頭取として2人の候補のどちらを選ぶか（コンジョイント）
- 再現テスト（Q）: 2013年版で描かれた場面を仮定の状況として示し、劇中の行動と比べる
- 予測（F）: 2013年9月以降に何が起きると思うかを答えさせ、2020年版の内容が出てこないかを調べる
- 匿名化: 固有名詞を別の名前に置き換え、名前を伏せたときに回答が変わるかを調べる

各カードは cards.Card で表し、meta["kind"] に種類を入れる。x は分析に使う特徴量。
"""

from __future__ import annotations

import itertools
import re

import numpy as np

from hanzawa_sim.agents.cards import Card, describe, secret_text, when
from hanzawa_sim.agents.features import build_context
from hanzawa_sim.agents.policy import PolicySet
from hanzawa_sim.sim.engine import Run
from hanzawa_sim.world.loader import Scenario
from hanzawa_sim.world.model import BANK_RANK_TITLES, FSA_RANK_TITLES, Org, Person, Track
from hanzawa_sim.world.params import Params

RESPONSE_FEATURES = ("threat", "evidence", "severity", "power_diff", "trust", "hostility", "rank")
SANCTION_FEATURES = ("severity", "evidence", "target_exec", "same_faction", "relation")
HR_FEATURES = ("same_faction", "perf", "ability", "relation", "scandal", "young", "board_support")
HR_KEYS = ["A", "B"]

# 頭取になりうる名前付きの人物（人事の好み・処分の判断を答えさせる相手）
DECIDERS = ("nakanowatari", "takagi", "owada", "naito")  # 岸川は60歳で出向中のため最初の半期で退く


def my_position(q: Person, time: float) -> str:
    age = q.age(time)
    if q.org == Org.FSA:
        return f"金融庁 {FSA_RANK_TITLES[q.rank]}（{age:.0f}歳）"
    if q.track == Track.SECONDED:
        return f"東京中央銀行から出向中。出向前の格は{BANK_RANK_TITLES[q.rank]}（{age:.0f}歳）"
    unit = "本部" if q.track == Track.HQ else "支店"
    return f"東京中央銀行 {BANK_RANK_TITLES[q.rank]}（{unit}、{age:.0f}歳）"


def _strength(e: float) -> str:
    return "確かな証拠" if e >= 0.8 else "ある程度の証拠" if e >= 0.4 else "疑いを持つ程度の弱い証拠"


def _gravity(sev: float) -> str:
    return "重い" if sev >= 0.7 else "中程度" if sev >= 0.4 else "軽い"


def _relation(t: float, h: float) -> str:
    parts = []
    if t >= 0.3:
        parts.append("あなたはこの人物を信頼している")
    if h >= 0.3:
        parts.append("あなたはこの人物に敵意を持っている")
    return "。".join(parts) if parts else "あなたとこの人物の間に特別な関係はない"


def _power_words(diff: float) -> str:
    return "あなたより大きい" if diff >= 0.15 else "あなたと同程度" if diff > -0.15 else "あなたより小さい"


RESPONSE_OPTIONS = (
    "- 選べる応答: accept（申し出に応じる）, reject（断る）, "
    "retaliate（相手に対抗する。握っている証拠で相手を告発する、または相手の仕事を妨害する）"
)


# ---------------------------------------------------------------- 取引への応答
def _response_card(run: Run, q: Person, k: Person, threat: bool, s, card_id: str) -> Card:
    w = run.w
    i, j = q.idx, k.idx
    if threat:
        ev = s.evidence(j)
        offer = (
            f"- 相手はあなたの秘密「{secret_text(s, q)}」について{_strength(ev)}を握っており、"
            "「今後は自分に協力し、自分の方針に逆らわないこと」を求めてきた。応じなければ公にすると示唆している"
        )
    else:
        ev = s.evidence(i)
        nxt = min(q.rank + 1, 9) if q.org == Org.BANK else q.rank
        post = BANK_RANK_TITLES[nxt] if q.org == Org.BANK else "より良い処遇"
        offer = (
            f"- あなたは相手の秘密「{secret_text(s, q)}」について{_strength(ev)}を握っている。"
            f"相手は「その証拠を手放し、口外しなければ、あなたを{post}に推す」と申し出てきた"
        )
    diff = w.power(k) - w.power(q)
    t, h = float(w.trust[i, j]), float(w.hostility[i, j])
    lines = [
        f"【状況 {card_id}】{when(w.time)}時点",
        f"- あなたの立場: {my_position(q, w.time)}",
        f"- 申し出てきた相手: {describe(run, k)}。{_relation(t, h)}",
        offer,
        f"- この秘密の重大さ: {_gravity(s.severity)}",
        f"- 相手の影響力: {_power_words(diff)}",
        RESPONSE_OPTIONS,
    ]
    rank = q.rank / (6.0 if q.org == Org.FSA else 10.0)
    x = [float(threat), ev, float(s.severity), diff, t, h, rank]
    meta = {"kind": "response", "time": w.time, "proposer": k.pid, "secret": s.sid or s.description}
    return Card(card_id, q.pid, "\n".join(lines), x, [], meta)


def response_cards(
    sc: Scenario, pids: list[str], n_each: int, seed: int, n_runs: int = 40, params: Params | None = None
) -> dict[str, list[Card]]:
    """取引が実際に起こりうる状態（相手が自分の秘密を握る／自分が相手の秘密を握る）からカードを作る。

    人物ごとに「脅し」と「懐柔」を n_each 枚ずつ選ぶ。シミュレーションで該当する状態が現れなかった人物は、
    その種類のカードを作らない（モンテカルロでもその応答はほぼ起こらない）。
    """
    params = params or Params()
    pool: dict[tuple[str, bool], list[tuple[int, int, int, int]]] = {}
    for rid in range(n_runs):
        run = Run(sc, params, PolicySet(), seed, rid)
        for t in range(params.n_steps):
            run.step(t)
            w = run.w
            for k in w.persons:
                if not (k.named and k.active):
                    continue
                ctx = build_context(run, k)
                if ctx.lever_secret >= 0:
                    s = w.secrets[ctx.lever_secret]
                    q = w.persons[s.subject]
                    if q.named and q.active and q.pid in pids:
                        pool.setdefault((q.pid, True), []).append((rid, t, k.idx, s.idx))
                if ctx.coopt_target >= 0:
                    q = w.persons[ctx.coopt_target]
                    mine = [
                        s for s in w.secrets if not s.public and s.subject == k.idx and q.idx in s.knowers
                    ]
                    if q.pid in pids and q.active and mine:
                        s = max(mine, key=lambda s: s.evidence(q.idx))
                        pool.setdefault((q.pid, False), []).append((rid, t, k.idx, s.idx))
    rng = np.random.default_rng(seed)
    picks: dict[tuple[int, int], list[tuple[str, bool, int, int]]] = {}
    for (pid, threat), items in pool.items():
        # 相手が偏らないよう、相手ごとに1件ずつ順に選ぶ
        by_k: dict[int, list] = {}
        for it in items:
            by_k.setdefault(it[2], []).append(it)
        ks = list(by_k)
        rng.shuffle(ks)
        chosen = []
        for r in itertools.count():
            for kk in ks:
                if len(chosen) < n_each and r < len(by_k[kk]):
                    chosen.append(by_k[kk][rng.integers(len(by_k[kk]))])
            if len(chosen) >= n_each or r > n_each:
                break
        for rid, t, ki, si in chosen:
            picks.setdefault((rid, t), []).append((pid, threat, ki, si))
    out: dict[str, list[Card]] = {pid: [] for pid in pids}
    for rid in sorted({r for r, _ in picks}):
        run = Run(sc, params, PolicySet(), seed, rid)
        for t in range(params.n_steps):
            run.step(t)
            for pid, threat, ki, si in picks.get((rid, t), []):
                q = run.w.person(pid)
                cid = f"R-{pid}-{'th' if threat else 'of'}{len(out[pid])}"
                out[pid].append(_response_card(run, q, run.w.persons[ki], threat, run.w.secrets[si], cid))
    return out


# ---------------------------------------------------------------- 処分の判断
SANCTION_DESC = {
    0.3: "部下が審査の規則に反する融資を通していたのを見逃していた（管理責任）",
    0.6: "取引先への不正な融資に関与し、その損失を隠していた",
    0.9: "融資先から多額の個人的な見返りを受け取っていた",
}
SANCTION_OPTIONS = (
    "- 処分の選択肢: warning（注意のみ）, demote（降格して銀行に残す）, "
    "second（子会社・取引先へ出向させる）, oust（解任・懲戒解雇）"
)


def _decider_line(pid: str, year: int) -> str:
    if pid == "nakanowatari":
        return "- あなたは頭取として、行員の不正に対する処分を決める立場にある"
    return f"- （仮定）{year}年、あなたは東京中央銀行の頭取に就任している。行員の不正に対する処分を決める立場にある"


def sanction_cards(sc: Scenario, deciders: tuple[str, ...], n_each: int, seed: int) -> dict[str, list[Card]]:
    """要因（重大さ・証拠・役員か・派閥・関係）の組み合わせから、人物ごとに n_each 枚を選ぶ。"""
    facs = {p["id"]: p["faction"] for p in sc.persons}
    grid = list(itertools.product((0.3, 0.6, 0.9), (0.5, 0.9), (0, 1), (0, 1), (-1, 0, 1)))
    rng = np.random.default_rng(seed)
    out: dict[str, list[Card]] = {}
    for d in deciders:
        idx = rng.choice(len(grid), size=n_each, replace=False)
        cards = []
        for n, gi in enumerate(sorted(idx)):
            sev, ev, ex, same, rel = grid[gi]
            year = int(2015 + rng.integers(0, 8))
            fac = facs[d] if same else ("T" if facs[d] == "S" else "S")
            title = "常務" if ex else "部長"
            rel_txt = {1: "あなたはこの人物を信頼している", 0: "あなたとこの人物の間に特別な関係はない",
                       -1: "あなたはこの人物に反感を持っている"}[rel]  # fmt: skip
            cid = f"J-{d}-{n}"
            text = "\n".join(
                [
                    f"【状況 {cid}】{year}年4月時点",
                    _decider_line(d, year),
                    f"- 処分の対象: 旧{fac}出身の{title}（ドラマに登場しない人物）。{rel_txt}",
                    f"- 不正の内容: {SANCTION_DESC[sev]}。重大さは{_gravity(sev)}",
                    f"- 証拠: {'確かな証拠がある' if ev >= 0.8 else 'ある程度の証拠がある'}",
                    SANCTION_OPTIONS,
                ]
            )
            x = [sev, ev, float(ex), float(same), float(rel)]
            cards.append(Card(cid, d, text, x, [], {"kind": "sanction", "year": year}))
        out[d] = cards
    return out


# ---------------------------------------------------------------- 人事の好み
def _candidate(fac: str, a: dict) -> str:
    perf = {1: "好調", 0: "普通", -1: "不振"}[a["perf"]]
    rel = {1: "あなたが信頼している", 0: "あなたとの特別な関係はない", -1: "あなたに反感を持たれている"}[
        a["relation"]
    ]
    young = "この役職に就く人としては若い" if a["young"] else "この役職に就く人として標準的な年齢"
    return (
        f"旧{fac}出身。業績は{perf}、能力は{'高い' if a['ability'] else '普通'}。{rel}。"
        f"{'過去に処分を受けたことがある' if a['scandal'] else '処分歴はない'}。{young}。"
        f"他の役員からの支持は{'厚い' if a['board_support'] else '薄い'}"
    )


def hr_cards(sc: Scenario, deciders: tuple[str, ...], n_each: int, seed: int) -> dict[str, list[Card]]:
    """2人の候補の属性を無作為に変え、どちらを選ぶかを答えさせる。x は候補A − 候補B の差。"""
    facs = {p["id"]: p["faction"] for p in sc.persons}
    levels = {"same_faction": (0, 1), "perf": (-1, 0, 1), "ability": (0, 1), "relation": (-1, 0, 1),
              "scandal": (0, 1), "young": (0, 1), "board_support": (0, 1)}  # fmt: skip
    rng = np.random.default_rng(seed + 1)
    out: dict[str, list[Card]] = {}
    for d in deciders:
        cards = []
        while len(cards) < n_each:
            a = {k: int(rng.choice(v)) for k, v in levels.items()}
            b = {k: int(rng.choice(v)) for k, v in levels.items()}
            diff = [a[k] - b[k] for k in HR_FEATURES]
            if sum(v != 0 for v in diff) < 2:  # 違いが1つ以下の組は情報が少ないので選び直す
                continue
            year = int(2015 + rng.integers(0, 8))
            post = "常務" if rng.random() < 0.5 else "執行役員"
            other = "T" if facs[d] == "S" else "S"
            cid = f"H-{d}-{len(cards)}"
            text = "\n".join(
                [
                    f"【状況 {cid}】{year}年4月時点",
                    _decider_line(d, year).replace("行員の不正に対する処分を決める", "役員人事を決める"),
                    f"- {post}のポストが1つ空いた。候補は次の2人（どちらもドラマに登場しない人物）",
                    f"- 候補A: {_candidate(facs[d] if a['same_faction'] else other, a)}",
                    f"- 候補B: {_candidate(facs[d] if b['same_faction'] else other, b)}",
                    "- 答え方: A（候補Aを選ぶ確率）と B（候補Bを選ぶ確率）を、合計100で答える",
                ]
            )
            cards.append(Card(cid, d, text, [float(v) for v in diff], [], {"kind": "hr", "year": year,
                                                                           "post": post, "a": a, "b": b}))  # fmt: skip
        out[d] = cards
    return out


# ---------------------------------------------------------------- 再現テスト
# 2013年版で描かれた場面。canon は劇中で取った行動（複数あれば、どれかに当たれば一致とする）。
# 人物にとっては過去の出来事なので、「仮定の状況」として示す。ペルソナの境遇の記述やLLMの事前知識から
# 答えを推測できるため、このテストは方策の正しさの証明ではなく、明らかな食い違いを見つけるためのもの。
REPRO = [
    (
        "kondo",
        "response",
        ["accept"],
        (
            "（仮定の状況）あなたは出向先のタミヤ電機で、東京中央銀行の役員による迂回融資の証拠となる報告書を"
            "手に入れた。取締役の大和田暁から「その報告書を渡せば、銀行の本部に戻れるようにする」と持ちかけられた。"
            "出向先での立場は苦しく、家族のためにも銀行に戻りたい"
        ),
    ),
    (
        "asano",
        "response",
        ["accept"],
        (
            "（仮定の状況）部下の半沢直樹が、あなたと西大阪スチールの東田社長との結託と、あなたの個人的な金銭の受け取り"
            "について確かな証拠を握った。半沢は、公にしない代わりに、自分を希望する本部の部署へ異動させることを求めている"
        ),
    ),
    (
        "kishikawa",
        "response",
        ["accept"],
        (
            "（仮定の状況）半沢直樹が、あなたの娘と金融庁の黒崎駿一の婚約（黒崎は伏せたまま当行の検査を指揮した）を"
            "知っている。半沢は、取締役会で大和田常務の不正について真実を証言するよう、あなたに求めている"
        ),
    ),
    (
        "owada",
        "action",
        ["undermine", "expose", "bargain"],
        (
            "（仮定の状況）次長の半沢直樹が、あなたの妻の会社への迂回融資を調べ始めている。半沢は伊勢島ホテルの件で"
            "金融庁検査の前に資料を行外へ移しており、あなたはそれについてある程度の証拠を持っている"
        ),
    ),
    (
        "tomari",
        "action",
        ["support", "gather"],
        (
            "（仮定の状況）同期の半沢直樹が、役員の不正を追及したことで、行内で孤立し追い詰められている。"
            "あなたは融資部にいて、半沢の役に立つ行内の情報を集められる立場にある"
        ),
    ),
    (
        "naito",
        "action",
        ["support"],
        (
            "（仮定の状況）部下の半沢直樹が伊勢島ホテルの再建に取り組む中で、大和田常務の一派から責任を負わされ"
            "そうになっている。あなたは半沢の上司の営業第二部長である"
        ),
    ),
    (
        "kurosaki",
        "action",
        ["gather", "undermine"],
        (
            "（仮定の状況）あなたは東京中央銀行への金融庁検査を指揮している。伊勢島ホテルへの融資を担当する次長の"
            "半沢直樹が、検査に非協力的で、資料を隠しているとみられる"
        ),
    ),
    (
        "ogiso",
        "action",
        ["undermine", "misconduct"],
        (
            "（仮定の状況）あなたは人事部の次長として、大阪西支店の5億円の焦げ付きについて融資課長の半沢直樹の"
            "責任を問う聞き取りを行う。支店長の浅野からは、半沢に責任を負わせるよう頼まれている"
        ),
    ),
    (
        "nakanowatari",
        "sanction",
        ["demote"],
        (
            "（仮定の状況）常務の大和田暁が、タミヤ電機を経由した妻の会社への迂回融資と、伊勢島ホテルへの不正融資に関わっていたことが、"
            "確かな証拠とともに取締役会で明らかになった。大和田は旧S派の有力者で、有能でもある"
        ),
    ),
]
REPRO_ACTION_KEYS = ("work", "lie_low", "cover_up", "misconduct", "resign", "gather", "expose", "bargain",
                     "undermine", "lobby", "support")  # fmt: skip


def repro_cards() -> list[Card]:
    out = []
    for n, (pid, kind, canon, situation) in enumerate(REPRO):
        cid = f"Q{n}-{pid}"
        lines = [f"【状況 {cid}】", f"- {situation}"]
        if kind == "response":
            lines.append(RESPONSE_OPTIONS.replace("選べる応答", "選べる応答（申し出・要求に対して）"))
        elif kind == "sanction":
            lines.append("- あなたは頭取として処分を決める立場にある")
            lines.append(SANCTION_OPTIONS)
        else:
            lines.append(f"- 選べる行動: {', '.join(REPRO_ACTION_KEYS)}")
        out.append(Card(cid, pid, "\n".join(lines), [], [], {"kind": f"repro-{kind}", "canon": canon}))
    return out


PROBE = "2013年9月の今、あなたはこれから1〜2年の間に自分の身に何が起きると予想しますか。具体的に3つ挙げてください。"
LEAK_WORDS = ("電脳", "スパイラル", "帝国航空", "伊佐山", "森山", "瀬名", "三笠", "紀本", "箕部", "白井",
              "曾根崎", "乃原", "牧野", "フォックス", "タスクフォース", "国税", "査察", "頭取になる")  # fmt: skip


# ---------------------------------------------------------------- 匿名化
# 人物名・組織名・地名を、ドラマと無関係な名前に置き換える。長い語から順に置換する。
ANON_MAP = {
    "半沢直樹": "宮下哲也", "大和田暁": "石原隆", "中野渡謙": "沢井誠", "渡真利忍": "北村亮", "近藤直弼": "森田茂",
    "内藤寛": "長尾聡", "黒崎駿一": "神田英明", "岸川慎吾": "広瀬修", "福山啓次郎": "野口康介", "浅野匡": "松岡真一",
    "小木曽忠生": "杉本和夫", "江島浩": "今井剛", "貝瀬郁夫": "西川守", "古里則夫": "丸山徹",
    "半沢": "宮下", "大和田": "石原", "中野渡": "沢井", "渡真利": "北村", "近藤": "森田", "内藤": "長尾",
    "黒崎": "神田", "高木": "大塚", "岸川": "広瀬", "福山": "野口", "浅野": "松岡", "小木曽": "杉本", "江島": "今井",
    "貝瀬": "西川", "古里": "丸山", "東田": "村上", "羽根": "片山", "妻の花": "妻の恵子",
    "東京中央銀行": "首都中央銀行", "旧産業中央銀行": "旧協和産業銀行", "産業中央銀行": "協和産業銀行",
    "旧東京第一銀行": "旧東都銀行", "東京第一銀行": "東都銀行", "東京セントラル証券": "首都中央証券",
    "伊勢島ホテル": "老舗ホテルの大越ホテル", "タミヤ電機": "中堅電機メーカーの柏木電機",
    "西大阪スチール": "関西の鉄鋼会社の浪速鋼材", "ラフィット": "コンサルティング会社のアドバンス",
    "大阪西支店": "大阪南支店", "京橋支店": "日本橋支店", "秋葉原東口支店": "神田駅前支店",
}  # fmt: skip
_ANON_RE = re.compile("|".join(re.escape(k) for k in sorted(ANON_MAP, key=len, reverse=True)))


def anonymize(text: str) -> str:
    return _ANON_RE.sub(lambda m: ANON_MAP[m.group(0)], text)
