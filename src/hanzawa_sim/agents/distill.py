"""LLM エージェントの回答から方策モデルを学習する（蒸留、PLAN.md 4.8 節）。

行動: 人物 g の多項ロジット P(a | x) ∝ exp(W_g · [1, x])。選べない行動は確率0とする。
  W_g = B + D_g（人物共通の部分 B と人物別の差分 D_g）。LLMが答えた確率分布との交差エントロピーに
  L2 正則化を加えて最小化する。D_g の正則化でカードの少ない人物を共通部分に寄せる。
取引への応答・処分の判断・人事の好みも同じ形で、クラス数と特徴だけが違う。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from hanzawa_sim.agents.elicit import RESPONSE_KEYS, SANCTION_KEYS, normalize_action
from hanzawa_sim.agents.special_cards import HR_KEYS
from hanzawa_sim.world.model import N_ACTIONS

KEYS = {"response": RESPONSE_KEYS, "sanction": SANCTION_KEYS, "hr": HR_KEYS}


# ---------------------------------------------------------------- 回答の読み込み
@dataclass
class Answer:
    job_id: str
    model: str
    schema: str
    pid: str
    card_id: str
    variant: str
    retest: bool
    split: str
    x: np.ndarray
    mask: np.ndarray
    p: np.ndarray
    reason: str
    other_mass: float = 0.0
    masked_mass: float = 0.0


def load_answers(out_dir: Path) -> list[Answer]:
    jobs = {j["id"]: j for j in json.loads((out_dir / "jobs.json").read_text(encoding="utf-8"))}
    latest: dict[str, dict] = {}
    for line in (out_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r["is_error"] and r.get("output") is not None and r["tag"] in jobs:
            latest[r["tag"]] = r
    out = []
    for jid, r in latest.items():
        j = jobs[jid]
        schema, card = j["schema"], j["card"]
        if schema == "probe" or card is None:
            continue
        ans = (r["output"].get("answers") or [None])[0]
        if not ans or "probs" not in ans:
            continue
        kind = card["meta"].get("kind", "action")
        if schema == "action":
            mask = np.array(card["mask"], dtype=bool)
            try:
                nd = normalize_action(ans, card["mask"])
            except ValueError:
                continue
            if nd["dist"] is None:
                continue
            p, om, mm = np.array(nd["dist"]), nd["other_mass"], nd["masked_mass"]
        else:
            keys = KEYS[schema]
            raw = np.array([float(ans["probs"].get(k, 0)) for k in keys])
            if raw.sum() <= 0:
                continue
            p, mask, om, mm = raw / raw.sum(), np.ones(len(keys), dtype=bool), 0.0, 0.0
        out.append(
            Answer(
                job_id=jid,
                model=j["model"],
                schema=schema if not kind.startswith("repro") else "repro-" + schema,
                pid=j["pid"],
                card_id=card["card_id"],
                variant=j["variant"],
                retest=jid.endswith("#retest"),
                split=card["meta"].get("split", "train"),
                x=np.array(card["x"], dtype=float),
                mask=mask,
                p=p,
                reason=ans.get("reason", ""),
                other_mass=om,
                masked_mass=mm,
            )
        )
    return out


# ---------------------------------------------------------------- 学習
@dataclass
class SoftmaxModel:
    """W[g] = B + D[g]。groups[g] は人物ID。未知の人物は B を使う。"""

    features: list[str]
    classes: list[str]
    groups: list[str]
    B: np.ndarray  # (K, F+1)
    D: np.ndarray  # (G, K, F+1)
    lam: tuple[float, float]

    def W(self, pid: str) -> np.ndarray:
        return self.B + self.D[self.groups.index(pid)] if pid in self.groups else self.B

    def predict(self, pid: str, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        z = self.W(pid) @ np.concatenate([[1.0], x])
        if mask is not None:
            z = np.where(mask, z, -np.inf)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def to_json(self) -> dict:
        return {
            "features": self.features,
            "classes": self.classes,
            "lam": list(self.lam),
            "shared": self.B.round(5).tolist(),
            "person": {g: (self.B + self.D[k]).round(5).tolist() for k, g in enumerate(self.groups)},
        }


def _objective(theta, X1, P, M, gi, G, K, F, lam0, lam1, lam_int):
    n = len(X1)
    B = theta[: K * F].reshape(K, F)
    D = theta[K * F :].reshape(G, K, F)
    W = B[None] + D
    Z = np.einsum("nf,nkf->nk", X1, W[gi])
    Z = np.where(M, Z, -1e30)
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z) * M
    Q = E / E.sum(1, keepdims=True)
    logQ = np.log(np.where(M, Q, 1.0) + 1e-300)
    nll = -(P * logQ).sum() / n
    R = (Q - P) / n
    gW = np.einsum("nk,nf->nkf", R, X1)
    gD = np.zeros((G, K, F))
    np.add.at(gD, gi, gW)
    gB = gD.sum(0)
    reg_B = np.ones((K, F)) * lam0
    reg_B[:, 0] = lam_int
    loss = nll + 0.5 * (reg_B * B**2).sum() + 0.5 * lam1 * (D**2).sum()
    gB = gB + reg_B * B
    gD = gD + lam1 * D
    return loss, np.concatenate([gB.ravel(), gD.ravel()])


def fit_softmax(
    X: np.ndarray,
    P: np.ndarray,
    M: np.ndarray,
    pids: list[str],
    features: list[str],
    classes: list[str],
    lam0: float,
    lam1: float,
    lam_int: float = 1e-3,
) -> SoftmaxModel:
    groups = sorted(set(pids))
    gi = np.array([groups.index(p) for p in pids])
    X1 = np.hstack([np.ones((len(X), 1)), X])
    G, K, F = len(groups), P.shape[1], X1.shape[1]
    theta0 = np.zeros(K * F + G * K * F)
    res = minimize(
        _objective,
        theta0,
        args=(X1, P, M, gi, G, K, F, lam0, lam1, lam_int),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 3000},
    )
    B = res.x[: K * F].reshape(K, F)
    D = res.x[K * F :].reshape(G, K, F)
    return SoftmaxModel(features, classes, groups, B, D, (lam0, lam1))


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def evaluate(model: SoftmaxModel, data: list[Answer]) -> float:
    return (
        float(np.mean([tv(model.predict(a.pid, a.x, a.mask), a.p) for a in data])) if data else float("nan")
    )


def arrays(data: list[Answer]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X = np.array([a.x for a in data])
    P = np.array([a.p for a in data])
    M = np.array([a.mask for a in data])
    return X, P, M, [a.pid for a in data]


def cv_fit(
    data: list[Answer],
    features: list[str],
    classes: list[str],
    lam0_grid: tuple[float, ...],
    lam1_grid: tuple[float, ...],
    k: int = 5,
    seed: int = 0,
) -> tuple[SoftmaxModel, dict]:
    """k 分割の交差検証で正則化の強さを選び、全データで学習し直す。"""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(data)) % k
    scores = {}
    for l0 in lam0_grid:
        for l1 in lam1_grid:
            errs = []
            for f in range(k):
                tr = [a for a, z in zip(data, fold, strict=True) if z != f]
                te = [a for a, z in zip(data, fold, strict=True) if z == f]
                m = fit_softmax(*arrays(tr), features, classes, l0, l1)
                errs.append(evaluate(m, te))
            scores[(l0, l1)] = float(np.mean(errs))
    best = min(scores, key=lambda key: scores[key])
    model = fit_softmax(*arrays(data), features, classes, *best)
    return model, {f"{a}/{b}": round(v, 4) for (a, b), v in scores.items()} | {"best": list(best)}


def person_mean_baseline(train: list[Answer], test: list[Answer]) -> float:
    """比較用: 状況を見ずに、人物ごとの平均分布（選べない行動を除いて正規化）を返すモデル。"""
    means = {}
    for pid in {a.pid for a in train}:
        means[pid] = np.mean([a.p for a in train if a.pid == pid], axis=0)
    glob = np.mean([a.p for a in train], axis=0)
    errs = []
    for a in test:
        m = np.where(a.mask, means.get(a.pid, glob), 0.0)
        m = m / m.sum() if m.sum() > 0 else np.where(a.mask, 1.0, 0.0) / a.mask.sum()
        errs.append(tv(m, a.p))
    return float(np.mean(errs)) if errs else float("nan")


__all__ = ["N_ACTIONS", "Answer", "SoftmaxModel", "cv_fit", "evaluate", "fit_softmax", "load_answers", "tv"]
