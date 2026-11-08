"""世界モデルの型定義（人物、秘密、世界の状態）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Org(IntEnum):
    BANK = 0
    FSA = 1


class Track(IntEnum):
    HQ = 0  # 本部
    BRANCH = 1  # 支店
    SECONDED = 2  # 出向
    TRANSFERRED = 3  # 転籍（銀行には戻らない）
    RETIRED = 4  # 退職・退任
    OUSTED = 5  # 失脚（解任・懲戒・逮捕）
    EXTERNAL = 6  # 外部組織（金融庁）


ACTIVE_TRACKS = frozenset({Track.HQ, Track.BRANCH, Track.SECONDED, Track.EXTERNAL})
IN_BANK = frozenset({Track.HQ, Track.BRANCH})
TRACK_JA = {
    Track.HQ: "本部",
    Track.BRANCH: "支店",
    Track.SECONDED: "出向",
    Track.TRANSFERRED: "転籍",
    Track.RETIRED: "退職・退任",
    Track.OUSTED: "失脚",
    Track.EXTERNAL: "外部",
}


class Action(IntEnum):
    WORK = 0  # 業務専念
    LIE_LOW = 1  # 静観
    COVER_UP = 2  # 隠蔽
    MISCONDUCT = 3  # 不正
    RESIGN = 4  # 退職
    GATHER = 5  # 証拠収集
    EXPOSE = 6  # 告発
    BARGAIN = 7  # 取引
    UNDERMINE = 8  # 妨害
    LOBBY = 9  # 根回し
    SUPPORT = 10  # 支援


N_ACTIONS = len(Action)
ACTION_JA = {
    Action.WORK: "業務専念",
    Action.LIE_LOW: "静観",
    Action.COVER_UP: "隠蔽",
    Action.MISCONDUCT: "不正",
    Action.RESIGN: "退職",
    Action.GATHER: "証拠収集",
    Action.EXPOSE: "告発",
    Action.BARGAIN: "取引",
    Action.UNDERMINE: "妨害",
    Action.LOBBY: "根回し",
    Action.SUPPORT: "支援",
}


class Response(IntEnum):
    """取引を持ちかけられた側の応答。"""

    ACCEPT = 0
    REJECT = 1
    RETALIATE = 2


BANK_RANK_TITLES = {
    10: "頭取",
    9: "副頭取",
    8: "専務",
    7: "常務",
    6: "取締役",
    5: "執行役員",
    4: "部長・主要支店長",
    3: "次長・副支店長",
    2: "調査役・課長",
    1: "課長代理以下",
}
FSA_RANK_TITLES = {1: "検査官", 2: "主任検査官", 3: "統括検査官", 4: "主任統括検査官", 5: "課長", 6: "審議官"}
TOUDORI_RANK = 10
EXEC_MIN_RANK = 5
EXEC_CAPACITY = {10: 1, 9: 2, 8: 3, 7: 6, 6: 8, 5: 12}
BANK_MAX_RANK = 10
FSA_MAX_RANK = 6

# 乱数表の添字の上限（人物・秘密とも）。超えた場合はエラーにする。
MAX_ENTITIES = 256


@dataclass(slots=True)
class Person:
    idx: int
    pid: str
    name: str
    named: bool  # True: 名前付き人物（LLMエージェント） / False: NPC
    faction: str  # "S" / "T" / "EXT"
    birth: float
    org: Org
    rank: int
    track: Track
    title: str
    ability: float
    traits: dict[str, float]
    perf: float = 0.0  # 直近の業績（おおむね -3〜3）
    reputation: float = 0.0  # 評判
    scandal: float = 0.0  # 処分歴の累積
    attacked: bool = False  # この半期に攻撃を受けたか
    attacked_last: bool = False  # 前の半期に攻撃を受けたか（行動選択の特徴に使う）
    rank_since: float = 0.0  # 現在のランクに就いた時期（最低在任年数の判定に使う）

    def age(self, time: float) -> float:
        return time - self.birth

    @property
    def active(self) -> bool:
        return self.track in ACTIVE_TRACKS

    @property
    def in_bank(self) -> bool:
        return self.org == Org.BANK and self.track in IN_BANK

    @property
    def is_exec(self) -> bool:
        return self.in_bank and self.rank >= EXEC_MIN_RANK

    def trait(self, key: str, default: float = 0.5) -> float:
        return self.traits.get(key, default)


@dataclass(slots=True)
class Secret:
    idx: int
    sid: str
    subject: int  # 当事者の人物添字
    accomplices: tuple[int, ...]
    severity: float
    knowers: dict[int, float]  # 人物添字 → 証拠の強さ（0〜1）
    created: float
    public: bool = False
    sanctioned: bool = False
    description: str = ""
    sources: dict[int, list[int]] = field(default_factory=dict)  # 人物添字 → 証拠を強めたイベントID

    def evidence(self, holder: int) -> float:
        return self.knowers.get(holder, 0.0)


@dataclass
class World:
    time: float
    step: int
    persons: list[Person]
    secrets: list[Secret]
    trust: np.ndarray  # trust[i, j]: i が j を信頼する度合い
    hostility: np.ndarray  # hostility[i, j]: i の j への敵意
    debt: np.ndarray  # debt[i, j]: i が j に負っている借り
    pid_index: dict[str, int]
    npc_serial: int = 0

    def person(self, pid: str) -> Person:
        return self.persons[self.pid_index[pid]]

    def toudori(self) -> Person | None:
        for q in self.persons:
            if q.in_bank and q.rank == TOUDORI_RANK:
                return q
        return None

    def occupancy(self, rank: int) -> int:
        return sum(1 for q in self.persons if q.in_bank and q.rank == rank)

    def power(self, q: Person) -> float:
        """行内での影響力（0〜1）。出向中は半分、金融庁は規制当局としての固定的な影響力を持つ。"""
        if not q.active:
            return 0.0
        if q.org == Org.FSA:
            return 0.3 + 0.1 * q.rank
        if q.track == Track.SECONDED:
            return q.rank / 20.0
        return q.rank / 10.0
