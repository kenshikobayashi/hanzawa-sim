"""初期状態の YAML を読み込み、試行ごとの World を組み立てる。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from hanzawa_sim.world.model import (
    EXEC_CAPACITY,
    MAX_ENTITIES,
    Org,
    Person,
    Secret,
    Track,
    World,
)
from hanzawa_sim.world.params import Params

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "world"

# 最終回時点で空いている役員ポスト（大和田の降格で空いた常務の席）
INITIAL_VACANCIES = {7: 1}

# NPC の年齢の範囲（ランク別）。各ランクの退任年齢より若くする
NPC_AGE_RANGE = {
    9: (59.0, 63.0),
    8: (57.0, 62.0),
    7: (54.0, 61.0),
    6: (52.0, 59.0),
    5: (50.0, 58.0),
    4: (45.0, 52.0),
}


@dataclass
class Scenario:
    start_time: float
    persons: list[dict]
    secrets: list[dict]
    private_facts: list[dict]
    edges: list[dict]


def load_scenario(data_dir: Path = DATA_DIR) -> Scenario:
    persons_doc = yaml.safe_load((data_dir / "persons.yaml").read_text(encoding="utf-8"))
    secrets_doc = yaml.safe_load((data_dir / "secrets.yaml").read_text(encoding="utf-8"))
    relations_doc = yaml.safe_load((data_dir / "relations.yaml").read_text(encoding="utf-8"))
    return Scenario(
        start_time=float(persons_doc["start_time"]),
        persons=persons_doc["persons"],
        secrets=secrets_doc["secrets"],
        private_facts=secrets_doc.get("private_facts", []),
        edges=relations_doc["edges"],
    )


def new_npc(w: World, rank: int, track: Track, u: np.ndarray, age_range: tuple[float, float]) -> Person:
    """乱数8個（u[0..7]）から NPC を1人作って World に加える。"""
    idx = len(w.persons)
    if idx >= MAX_ENTITIES:
        raise RuntimeError(f"人物数が上限 {MAX_ENTITIES} を超えた")
    serial = w.npc_serial
    w.npc_serial += 1
    lo, hi = age_range
    age = lo + (hi - lo) * float(u[1])
    ability = 0.45 + 0.45 * float(u[2])  # 部長級以上は選抜済みの層
    q = Person(
        idx=idx,
        pid=f"npc{serial:03d}",
        name=f"行員{serial:03d}",
        named=False,
        faction="S" if u[0] < 0.5 else "T",
        birth=w.time - age,
        org=Org.BANK,
        rank=rank,
        track=track,
        title="",
        ability=ability,
        traits={
            "ambition": 0.3 + 0.5 * float(u[3]),
            "aggression": 0.3,
            "integrity": 0.4 + 0.5 * float(u[4]),
            "loyalty": 0.5,
            "grudge": 0.3,
            "info_skill": 1.0,
            "faction_bias": 0.3 + 0.6 * float(u[5]),
            "cronyism": 0.3 + 0.5 * float(u[6]),
            "org_protect": 0.3 + 0.5 * float(u[7]),
        },
        perf=(ability - 0.5) * 2 * 0.5,
        rank_since=w.time - 3.0 * float(u[1]),
    )
    w.persons.append(q)
    return q


def build_world(sc: Scenario, params: Params, u_init: np.ndarray) -> World:
    """1試行分の World を作る。u_init は形状 (MAX_ENTITIES, 8) の乱数（NPC の生成に使う）。"""
    n = MAX_ENTITIES
    w = World(
        time=params.start_time,
        step=0,
        persons=[],
        secrets=[],
        trust=np.zeros((n, n), dtype=np.float32),
        hostility=np.zeros((n, n), dtype=np.float32),
        debt=np.zeros((n, n), dtype=np.float32),
        pid_index={},
    )
    for spec in sc.persons:
        idx = len(w.persons)
        ability = float(spec["ability"])
        w.persons.append(
            Person(
                idx=idx,
                pid=spec["id"],
                name=spec["name"],
                named=True,
                faction=spec["faction"],
                birth=float(spec["birth"]),
                org=Org.FSA if spec["org"] == "fsa" else Org.BANK,
                rank=int(spec["rank"]),
                track=Track[spec["track"]],
                title=spec.get("title", ""),
                ability=ability,
                traits=dict(spec["traits"]),
                perf=(ability - 0.5) * 2,
                scandal=float(spec.get("scandal", 0.0)),
                rank_since=float(spec.get("rank_since", params.start_time - 2.0)),
            )
        )
        w.pid_index[spec["id"]] = idx

    # 役員ポストの残りを NPC で埋める（頭取は中野渡）
    serial = 0
    for rank in sorted(EXEC_CAPACITY, reverse=True):
        if rank == 10:
            continue
        n_fill = EXEC_CAPACITY[rank] - w.occupancy(rank) - INITIAL_VACANCIES.get(rank, 0)
        for _ in range(n_fill):
            new_npc(w, rank, Track.HQ, u_init[serial], NPC_AGE_RANGE[rank])
            serial += 1
    # 部長級の NPC（執行役員ポストの競争相手）
    for _ in range(params.npc_pool_size):
        track = Track.HQ if u_init[serial][0] < 0.5 else Track.BRANCH
        new_npc(w, 4, track, u_init[serial], NPC_AGE_RANGE[4])
        serial += 1
    for q in w.persons:
        if not q.named:
            w.pid_index[q.pid] = q.idx

    # 秘密
    for spec in sc.secrets:
        idx = len(w.secrets)
        s = Secret(
            idx=idx,
            sid=spec["id"],
            subject=w.pid_index[spec["subject"]],
            accomplices=tuple(w.pid_index[a] for a in spec.get("accomplices", [])),
            severity=float(spec["severity"]),
            knowers={w.pid_index[k]: float(v) for k, v in spec["knowers"].items()},
            created=params.start_time - 0.25,
            public=bool(spec["public"]),
            sanctioned=bool(spec["sanctioned"]),
            description=spec.get("description", ""),
        )
        w.secrets.append(s)
        # 処分済みの不正は処分歴に反映する
        if s.sanctioned:
            w.persons[s.subject].scandal += 0.5 * s.severity
            for a in s.accomplices:
                w.persons[a].scandal += 0.35 * s.severity

    # 人間関係
    for e in sc.edges:
        i, j = w.pid_index[e["from"]], w.pid_index[e["to"]]
        w.trust[i, j] = float(e.get("trust", 0.0))
        w.hostility[i, j] = float(e.get("hostility", 0.0))
        w.debt[i, j] = float(e.get("debt", 0.0))
    return w
