"""Phase 3: LLM エージェントからの方策抽出（本番）。

パイロットの結果（PLAN.md）から、カードは1枚ずつ答えさせる。Sonnet を主とし、同じカードに Haiku でも
答えさせて、モデルによる違いを測る。

優先度（利用枠の上限に達したら、その時点で止まる。もう一度実行すると、終わっていない呼び出しから再開する）:
  tier 1: Sonnet  行動カード（学習用）、取引への応答、処分の判断、人事の好み
  tier 2: Sonnet  行動カード（検証用）、再回答、予測（情報漏れ）、再現テスト、名前を伏せた版
  tier 3: Haiku   tier 1・2 と同じもの

  uv run python scripts/elicit.py plan              # ジョブを作って件数を表示（呼び出さない）
  uv run python scripts/elicit.py show <job_id>     # プロンプトを表示
  uv run python scripts/elicit.py run --workers 4   # 実行
  uv run python scripts/elicit.py status            # 進み具合
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from hanzawa_sim.agents.cards import Card, sample_cards
from hanzawa_sim.agents.elicit import (
    ACTION_SCHEMA,
    PROBE_SCHEMA,
    RESPONSE_SCHEMA,
    SANCTION_SCHEMA,
    batch_prompt,
    prob_schema,
)
from hanzawa_sim.agents.llm import ClaudeCLI, UsageLimitError
from hanzawa_sim.agents.persona import load_personas, system_prompt
from hanzawa_sim.agents.special_cards import (
    DECIDERS,
    HR_KEYS,
    PROBE,
    anonymize,
    hr_cards,
    repro_cards,
    response_cards,
    sanction_cards,
)
from hanzawa_sim.world.loader import load_scenario
from hanzawa_sim.world.model import Action

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "elicitation" / "main"
SEED = 2026
MAIN = ["hanzawa", "owada", "nakanowatari", "tomari", "kondo", "naito", "kurosaki"]
N_MAIN, N_OTHER = 100, 40  # 行動カードの枚数（うち2割を検証用）
VAL_FRAC = 0.2
N_RETEST = 4  # 主要7人の再回答の枚数
ANON_PIDS = ["hanzawa", "owada", "kurosaki"]
SCHEMAS = {
    "action": ACTION_SCHEMA,
    "response": RESPONSE_SCHEMA,
    "sanction": SANCTION_SCHEMA,
    "hr": prob_schema(HR_KEYS),
    "probe": PROBE_SCHEMA,
}
ASK = {
    "response": "以下の状況で、各応答を選ぶ確率（合計100）と理由を答えてください。",
    "sanction": "以下の状況で、各処分を選ぶ確率（合計100）と理由を答えてください。",
    "hr": "以下の状況で、候補A・候補Bそれぞれを選ぶ確率（合計100）と理由を答えてください。",
}
CARD_ID_NOTE = "card_id にはカードの番号（【状況 …】の … の部分）をそのまま書いてください。"


def user_prompt(schema: str, card: Card | None) -> str:
    if schema == "probe":
        return PROBE
    assert card is not None
    if schema == "action":
        return batch_prompt([card])
    return ASK[schema] + CARD_ID_NOTE + "\n\n" + card.text


def card_dict(c: Card) -> dict:
    return {"card_id": c.card_id, "pid": c.pid, "text": c.text, "x": c.x, "mask": c.mask, "meta": c.meta}


def build_jobs() -> list[dict]:
    sc = load_scenario()
    pids = [p["id"] for p in sc.persons]
    others = [p for p in pids if p not in MAIN]
    t0 = time.time()
    action = sample_cards(sc, MAIN, N_MAIN, seed=SEED, n_runs=60, frac_far=0.5)
    action.update(sample_cards(sc, others, N_OTHER, seed=SEED, n_runs=60, frac_far=0.5))
    resp = response_cards(sc, pids, n_each=3, seed=SEED, n_runs=60)
    sanc = sanction_cards(sc, DECIDERS, n_each=8, seed=SEED)
    hr = hr_cards(sc, DECIDERS, n_each=10, seed=SEED)
    repro = repro_cards()
    print(f"カード生成 {time.time() - t0:.0f}秒")

    rng = np.random.default_rng(SEED)
    base: list[dict] = []  # (tier, schema, pid, variant, card)

    def add(
        tier: int, schema: str, pid: str, card: Card | None, variant: str = "named", tag: str = ""
    ) -> None:
        cid = card.card_id if card is not None else f"F-{pid}"
        base.append({"tier": tier, "schema": schema, "pid": pid, "variant": variant,
                     "key": f"{variant}/{cid}{tag}", "card": card_dict(card) if card else None})  # fmt: skip

    for pid, cs in action.items():
        n_val = round(len(cs) * VAL_FRAC)
        val = set(rng.choice(len(cs), size=n_val, replace=False).tolist())
        for k, c in enumerate(cs):
            c.meta["split"] = "val" if k in val else "train"
            add(1 if k not in val else 2, "action", pid, c)
        if pid in MAIN:
            train = [c for k, c in enumerate(cs) if k not in val]
            for c in train[:N_RETEST]:
                add(2, "action", pid, c, tag="#retest")
                if pid in ANON_PIDS:
                    ac = Card(c.card_id, c.pid, anonymize(c.text), c.x, c.mask, dict(c.meta))
                    add(2, "action", pid, ac, variant="anon")
    for pid, cs in resp.items():
        for c in cs:
            add(1, "response", pid, c)
    for pid, cs in sanc.items():
        for c in cs:
            add(1, "sanction", pid, c)
    for pid, cs in hr.items():
        for c in cs:
            add(1, "hr", pid, c)
    for c in repro:
        kind = c.meta["kind"].split("-")[1]
        if kind == "action":
            c.mask = [True] * len(Action)
        add(2, kind, c.pid, c)
    for pid in pids:
        add(2, "probe", pid, None)

    jobs = []
    for model in ("sonnet", "haiku"):
        for b in base:
            tier = 3 if model == "haiku" else b["tier"]
            jobs.append({**b, "tier": tier, "model": model, "id": f"{model}/{b['key']}"})
    ids = Counter(j["id"] for j in jobs)
    dup = [k for k, v in ids.items() if v > 1]
    assert not dup, dup[:5]
    return jobs


def load_jobs() -> list[dict]:
    path = OUT / "jobs.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    path.write_text(json.dumps(jobs, ensure_ascii=False, indent=0), encoding="utf-8")
    return jobs


def done_ids() -> set[str]:
    path = OUT / "calls.jsonl"
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r["is_error"] and r.get("output") is not None:
            out.add(r["tag"])
    return out


def system_prompts(jobs: list[dict]) -> dict[tuple[str, str], str]:
    sc = load_scenario()
    personas = load_personas()
    out = {}
    for pid, variant in {(j["pid"], j["variant"]) for j in jobs}:
        sp = system_prompt(sc, pid, personas)
        out[(pid, variant)] = anonymize(sp) if variant == "anon" else sp
    return out


def to_card(d: dict | None) -> Card | None:
    return None if d is None else Card(d["card_id"], d["pid"], d["text"], d["x"], d["mask"], d["meta"])


def summarize(jobs: list[dict], done: set[str]) -> None:
    tab: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for j in jobs:
        k = (j["tier"], j["model"], j["schema"])
        tab[k][1] += 1
        tab[k][0] += j["id"] in done
    for k in sorted(tab):
        print(f"  tier{k[0]} {k[1]:<6} {k[2]:<9} {tab[k][0]:>5}/{tab[k][1]}")
    print(f"  合計 {sum(v[0] for v in tab.values())}/{len(jobs)}")


def cost_so_far() -> dict[str, float]:
    path = OUT / "calls.jsonl"
    c: dict[str, float] = defaultdict(float)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            c[r["model"]] += float(r.get("cost_usd") or 0.0)
    return dict(c)


def run(jobs: list[dict], tiers: list[int], workers: int, limit: int | None) -> None:
    done = done_ids()
    sps = system_prompts(jobs)
    backends = {m: ClaudeCLI(m, OUT / "calls.jsonl", OUT / "prompts") for m in ("sonnet", "haiku")}
    stop = threading.Event()
    lock = threading.Lock()
    stats = Counter()
    t_start = time.time()
    budget = [limit if limit is not None else 10**9]

    def work(group: list[dict]) -> None:
        for j in group:
            if stop.is_set():
                return
            with lock:
                if budget[0] <= 0:
                    return
                budget[0] -= 1
            sp = sps[(j["pid"], j["variant"])]
            try:
                r = backends[j["model"]].call(j["id"], sp, user_prompt(j["schema"], to_card(j["card"])),
                                              SCHEMAS[j["schema"]])  # fmt: skip
            except UsageLimitError as e:
                print(f"\n利用枠の上限に達したため停止: {e}", flush=True)
                stop.set()
                return
            with lock:
                stats["ok" if r.ok else "fail"] += 1
                stats["cost"] += r.cost_usd
                n = stats["ok"] + stats["fail"]
                if n % 25 == 0 or not r.ok:
                    el = time.time() - t_start
                    print(f"  {n} 回（失敗 {stats['fail']}）{el / 60:.1f}分 換算 {stats['cost']:.2f}ドル"
                          + ("" if r.ok else f"  失敗: {j['id']} {r.error}"), flush=True)  # fmt: skip

    for tier in tiers:
        todo = [j for j in jobs if j["tier"] == tier and j["id"] not in done]
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for j in todo:  # 同じシステムプロンプトの呼び出しを1つのスレッドで続けて行い、キャッシュを効かせる
            groups[(j["model"], j["pid"], j["variant"])].append(j)
        order = sorted(groups.values(), key=len, reverse=True)
        print(f"tier {tier}: {len(todo)} 回 / {len(order)} グループ", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, order))
        if stop.is_set() or budget[0] <= 0:
            break
    print(f"\n今回: 成功 {stats['ok']} 失敗 {stats['fail']} 換算 {stats['cost']:.2f}ドル "
          f"{(time.time() - t_start) / 60:.1f}分")  # fmt: skip
    summarize(jobs, done_ids())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "show", "run", "status"])
    ap.add_argument("job_id", nargs="?")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tiers", default="1,2,3")
    ap.add_argument("--limit", type=int, default=None, help="この回で行う呼び出しの上限（試運転用）")
    args = ap.parse_args()
    jobs = load_jobs()
    if args.cmd == "plan":
        summarize(jobs, done_ids())
        per = Counter((j["pid"], j["schema"]) for j in jobs if j["model"] == "sonnet")
        for pid in dict.fromkeys(j["pid"] for j in jobs):
            print(f"  {pid:<13}", {s: n for (p, s), n in per.items() if p == pid})
    elif args.cmd == "show":
        j = next(j for j in jobs if j["id"] == args.job_id)
        print(system_prompts([j])[(j["pid"], j["variant"])])
        print("=" * 60)
        print(user_prompt(j["schema"], to_card(j["card"])))
    elif args.cmd == "status":
        summarize(jobs, done_ids())
        print("  換算額:", {k: round(v, 2) for k, v in cost_so_far().items()})
    else:
        run(jobs, [int(t) for t in args.tiers.split(",")], args.workers, args.limit)


if __name__ == "__main__":
    main()
