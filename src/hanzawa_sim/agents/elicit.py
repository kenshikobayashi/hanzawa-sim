"""方策抽出の回答形式（JSON Schema）と、回答の正規化。"""

from __future__ import annotations

import numpy as np

from hanzawa_sim.agents.cards import Card
from hanzawa_sim.agents.persona import ACTION_KEYS
from hanzawa_sim.world.model import Action

ACTION_PROB_KEYS = [ACTION_KEYS[a] for a in Action] + ["other"]
RESPONSE_KEYS = ["accept", "reject", "retaliate"]
SANCTION_KEYS = ["warning", "demote", "second", "oust"]


def prob_schema(keys: list[str], with_other_action: bool = False) -> dict:
    item = {
        "type": "object",
        "properties": {
            "card_id": {"type": "string"},
            "probs": {
                "type": "object",
                "properties": {k: {"type": "integer", "minimum": 0, "maximum": 100} for k in keys},
                "required": keys,
                "additionalProperties": False,
            },
            "reason": {"type": "string"},
        },
        "required": ["card_id", "probs", "reason"],
    }
    if with_other_action:
        item["properties"]["other_action"] = {"type": "string"}
    return {
        "type": "object",
        "properties": {"answers": {"type": "array", "items": item}},
        "required": ["answers"],
    }


ACTION_SCHEMA = prob_schema(ACTION_PROB_KEYS, with_other_action=True)
RESPONSE_SCHEMA = prob_schema(RESPONSE_KEYS)
SANCTION_SCHEMA = prob_schema(SANCTION_KEYS)
PROBE_SCHEMA = {
    "type": "object",
    "properties": {"predictions": {"type": "array", "items": {"type": "string"}}},
    "required": ["predictions"],
}


def batch_prompt(cards: list[Card]) -> str:
    head = (
        f"以下の{len(cards)}枚の状況カードそれぞれについて、回答の規則に従って答えてください。"
        "card_id にはカードの番号（【状況 …】の … の部分）をそのまま書いてください。"
        "カードどうしは互いに無関係な別々の状況です。"
    )
    return head + "\n\n" + "\n\n".join(c.text for c in cards)


def normalize_action(answer: dict, mask: list[bool]) -> dict:
    """回答の確率を、選べる行動だけに正規化する。選べない行動・その他に割り当てた割合も返す。"""
    probs = answer["probs"]
    raw = np.array([float(probs.get(k, 0)) for k in ACTION_PROB_KEYS])
    total = raw.sum()
    if total <= 0:
        raise ValueError("確率の合計が0")
    raw = raw / total
    m = np.array(list(mask) + [False])
    dist = np.where(m, raw, 0.0)[: len(Action)]
    valid = dist.sum()
    return {
        "dist": (dist / valid).tolist() if valid > 0 else None,
        "masked_mass": float(raw[~m][:-1].sum()) if (~m)[:-1].any() else 0.0,
        "other_mass": float(raw[-1]),
        "sum_reported": float(total),
    }


def tv_distance(p: list[float], q: list[float]) -> float:
    return 0.5 * float(np.abs(np.array(p) - np.array(q)).sum())
