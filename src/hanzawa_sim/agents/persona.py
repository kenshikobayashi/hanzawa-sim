"""人物ごとのシステムプロンプト（ペルソナ＋その人物が知っていること＋回答の規則）を組み立てる。

「知っていること」は data/world/*.yaml から機械的に作る。秘密はその人物が knowers に含まれる場合だけ書く
（エージェントの独立性 (b): 各人物は自分が知っている事実だけを受け取る）。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from hanzawa_sim.world.loader import DATA_DIR, Scenario
from hanzawa_sim.world.model import Action

PERSONA_PATH = DATA_DIR.parent / "personas" / "personas.yaml"

ACTION_KEYS = {
    Action.WORK: "work",
    Action.LIE_LOW: "lie_low",
    Action.COVER_UP: "cover_up",
    Action.MISCONDUCT: "misconduct",
    Action.RESIGN: "resign",
    Action.GATHER: "gather",
    Action.EXPOSE: "expose",
    Action.BARGAIN: "bargain",
    Action.UNDERMINE: "undermine",
    Action.LOBBY: "lobby",
    Action.SUPPORT: "support",
}
ACTION_DESC = {
    Action.WORK: "業務専念: 本業に集中して成果を上げる",
    Action.LIE_LOW: "静観: 目立つ行動を控え、様子を見る",
    Action.COVER_UP: "隠蔽: 自分の不正や弱みの証拠を消す、関係者に口止めする",
    Action.MISCONDUCT: "不正: 規則に反する手段で、目先の成果や個人的な利益を得る",
    Action.RESIGN: "退職: 銀行を辞める",
    Action.GATHER: "証拠収集: 最も敵対している相手の不正や弱みについて、証拠や情報を集める",
    Action.EXPOSE: "告発: 握っている相手の不正を、取締役会・頭取・金融庁などに訴えて公にする（失敗すると自分が処分を受けるおそれがある）",
    Action.BARGAIN: "取引: 握っている弱みを材料に相手に見返りを求める。または、自分の弱みを握る相手にポストなどの見返りを示して黙らせる",
    Action.UNDERMINE: "妨害: 敵の仕事を妨げる、失敗の責任を負わせる",
    Action.LOBBY: "根回し: 有力者に近づき、支持（後ろ盾）を得る",
    Action.SUPPORT: "支援: 味方の仕事を助け、味方の敵についての情報を渡す",
}

RULES = """## 回答の規則
- あなたは2013年9月時点のこの人物である。その後に起きたことは、状況カードに書かれた内容だけを知っている。ここにもカードにも書かれていない出来事を前提にしない
- 他人の秘密や内心のうち、あなたが知っていると書かれていないものは知らない。推測で補わない
- 正しい行動や望ましい行動ではなく、この人物の性格・価値観・利害から、実際に取りそうな行動を答える
- 状況カードごとに、各行動を選ぶ確率（0〜100の整数、合計100）を答える。カードで「選べない」とされた行動は0にする
- 選択肢に当てはまらない行動を取りそうなら other に確率を割り当て、other_action にその内容を書く
- reason には、この人物の視点で判断の理由を1〜2文で書く"""


_JA_SPACE = re.compile(r"(?<=[^\x00-\x7F]) (?=[^\x00-\x7F])")


def load_personas(path: Path = PERSONA_PATH) -> dict[str, dict[str, str]]:
    """YAML の折り返し（>-）で日本語の文中に入る空白を取り除いて読み込む。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["personas"]
    return {pid: {k: _JA_SPACE.sub("", v) for k, v in spec.items()} for pid, spec in raw.items()}


def _evidence_words(e: float) -> str:
    if e >= 0.8:
        return "確かな証拠を持っている"
    if e >= 0.4:
        return "ある程度の証拠を持っている"
    return "疑っているが、証拠は乏しい"


def _relation_words(e: dict) -> str:
    parts = []
    t, h, d = e.get("trust", 0.0), e.get("hostility", 0.0), e.get("debt", 0.0)
    if t > 0:
        parts.append("強く信頼している" if t >= 0.7 else "信頼している" if t >= 0.4 else "少し信頼している")
    if h > 0:
        parts.append(
            "強い敵意を持っている" if h >= 0.7 else "敵意を持っている" if h >= 0.4 else "やや反感を持っている"
        )
    if d > 0:
        parts.append("大きな借りがある" if d >= 0.5 else "借りがある")
    return "、".join(parts)


def knowledge_text(sc: Scenario, pid: str, include_s8: bool = True) -> str:
    names = {p["id"]: p["name"] for p in sc.persons}
    lines = ["## 主な人物の立場（2013年9月時点・周知の事実。その後の変化は状況カードに書かれる）"]
    lines += [f"- {p['name']}: {p['title']}" for p in sc.persons if p["id"] != pid]

    lines.append("\n## あなたが知っている不正・秘密")
    n0 = len(lines)
    for s in sc.secrets:
        subj, acc = s["subject"], s.get("accomplices", [])
        mine = pid == subj or pid in acc
        if s["public"]:
            who = "あなた自身" if pid == subj else names[subj]
            status = "処分済み" if s["sanctioned"] else "未処分"
            lines.append(f"- （公然の事実・{status}）{who}の不正: {s['description']}")
        elif pid in s["knowers"]:
            if pid == subj:
                lines.append(f"- あなた自身の秘密: {s['description']}")
            elif mine:
                lines.append(f"- あなたも関わった{names[subj]}の秘密: {s['description']}")
            else:
                lines.append(
                    f"- {names[subj]}の秘密: {s['description']}。あなたは{_evidence_words(s['knowers'][pid])}"
                )
    if len(lines) == n0:
        lines.append("- 特になし")
    for f in sc.private_facts:
        if pid in f["holders"] and (include_s8 or f["id"] != "S8"):
            lines.append(f"- {f['description']}")

    lines.append("\n## あなたから見た人間関係")
    rels = [e for e in sc.edges if e["from"] == pid]
    for e in rels:
        note = f"（{e['note']}）" if e.get("note") else ""
        lines.append(f"- {names[e['to']]}: {_relation_words(e)}{note}")
    if not rels:
        lines.append("- 特に強い関係はない")
    return "\n".join(lines)


def system_prompt(sc: Scenario, pid: str, personas: dict | None = None, include_s8: bool = True) -> str:
    personas = personas or load_personas()
    spec = next(p for p in sc.persons if p["id"] == pid)
    pr = personas[pid]
    actions = "\n".join(f"- {ACTION_KEYS[a]}（{ACTION_DESC[a]}）" for a in Action)
    return f"""あなたは{spec["name"]}です。以下の人物として、与えられた状況でこの人物が実際にどう行動するかを答えてください。
旧S＝旧産業中央銀行、旧T＝旧東京第一銀行（東京中央銀行は両行の合併で生まれた）。

## 人物像
{pr["profile"]}

## 2013年9月時点の境遇
{pr["situation"]}

## 価値観
{pr["values"]}

## この時点の目標
{pr["goals"]}

## 話し方
{pr["style"]}

{knowledge_text(sc, pid, include_s8)}

## 行動の選択肢（半期ごとに1つを選ぶ）
{actions}

{RULES}
"""
