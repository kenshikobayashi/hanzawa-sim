"""共通乱数法のための乱数表。

試行ごとに (半期, 用途, 対象, スロット) で添字を付けた一様乱数の表を最初に作る。
同じ対象・同じ用途の抽選には常に同じ位置の乱数を使うので、介入実験で1つのイベントを
変えても、無関係な抽選の乱数はずれない。
"""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np

from hanzawa_sim.world.model import MAX_ENTITIES


class P(IntEnum):
    ACTION = 0  # 行動の選択（人物別）
    OUTCOME = 1  # 行動の判定（人物別、slot 0〜5）
    PERF = 2  # 業績のノイズ（人物別、slot 0〜1）
    CASE = 3  # 案件の失敗（人物別、slot 0: 発生, 1: 重大度）
    DISCOVERY = 4  # 秘密の露見（秘密別、slot 0: 偶発, 1: 検査）
    SANCTION = 5  # 処分のノイズ（人物別、slot＝その半期の処分の回数）
    HR = 6  # 人事のガンベルノイズ（人物別、slot＝目標ランク−3）
    PROMO = 7  # 非役員の昇進など（人物別、slot 0: 昇進, 1: 復帰, 2: 年次出向, 3: 転籍, 4: 金融庁の昇進）
    GLOBAL = 8  # 全体のイベント（対象 0: 検査, 1: 派閥の不祥事, 2: 頭取の退任）
    NPC = 9  # NPCの生成（通し番号別）
    RESPONSE = 10  # 取引への応答（人物別）
    ORDER = 11  # 行動の処理順（人物別）
    INIT = 12  # 初期NPCの生成（通し番号別）


N_PURPOSES = len(P)
N_SLOTS = 8


def make_table(seed: int, run_id: int, n_steps: int) -> np.ndarray:
    """形状 (n_steps, N_PURPOSES, MAX_ENTITIES, N_SLOTS) の一様乱数（0, 1]。"""
    rng = np.random.default_rng([seed, run_id])
    return 1.0 - rng.random((n_steps, N_PURPOSES, MAX_ENTITIES, N_SLOTS), dtype=np.float32)


def normal(u1: float, u2: float) -> float:
    """2つの一様乱数から標準正規乱数を作る（Box–Muller）。"""
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def gumbel(u: float) -> float:
    return -math.log(-math.log(min(u, 1.0 - 1e-7)))
