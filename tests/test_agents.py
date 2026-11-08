import pytest

from hanzawa_sim.agents.cards import sample_cards
from hanzawa_sim.agents.elicit import ACTION_PROB_KEYS, normalize_action
from hanzawa_sim.agents.persona import load_personas, system_prompt
from hanzawa_sim.world.loader import load_scenario
from hanzawa_sim.world.model import Action


@pytest.fixture(scope="module")
def sc():
    return load_scenario()


@pytest.fixture(scope="module")
def prompts(sc):
    personas = load_personas()
    return {p["id"]: system_prompt(sc, p["id"], personas) for p in sc.persons}


def test_every_person_has_a_persona(sc):
    personas = load_personas()
    assert {p["id"] for p in sc.persons} == set(personas)


def test_secret_knowledge_is_isolated(prompts):
    """各人物のシステムプロンプトには、その人物が知っている秘密だけが入る。"""
    s2 = "東田と結託"  # 浅野の不正（知っているのは浅野と半沢）
    assert s2 in prompts["asano"] and s2 in prompts["hanzawa"]
    assert all(s2 not in prompts[p] for p in prompts if p not in ("asano", "hanzawa"))
    s3 = "婚約（私的関係）を伏せたまま"  # 黒崎の秘密（黒崎・岸川・半沢）
    assert all((s3 in prompts[p]) == (p in ("kurosaki", "kishikawa", "hanzawa")) for p in prompts)
    assert "あなた自身の秘密: 岸川の娘" in prompts["kurosaki"]


def test_private_fact_s8_only_for_nakanowatari(sc, prompts):
    s8 = "大和田派の報復から守り"
    assert s8 in prompts["nakanowatari"]
    assert all(s8 not in prompts[p] for p in prompts if p != "nakanowatari")
    assert s8 not in system_prompt(sc, "nakanowatari", include_s8=False)


def test_cards_render_with_masks(sc):
    cards = sample_cards(sc, ["hanzawa", "kurosaki"], n_per_person=4, seed=5, n_runs=3)
    for pid, cs in cards.items():
        assert len(cs) == 4, pid
        for c in cs:
            assert c.text.startswith(f"【状況 {c.card_id}】")
            assert "選べる行動" in c.text
            assert c.mask[Action.WORK]


def test_normalize_action_drops_masked_and_other():
    mask = [True] * len(Action)
    mask[Action.EXPOSE] = False
    probs = dict.fromkeys(ACTION_PROB_KEYS, 0)
    probs.update({"work": 50, "expose": 25, "other": 25})
    out = normalize_action({"probs": probs}, mask)
    assert out["dist"][Action.WORK] == pytest.approx(1.0)
    assert out["masked_mass"] == pytest.approx(0.25)
    assert out["other_mass"] == pytest.approx(0.25)
