"""世界モデルのパラメータ。

すべて仮置き。Phase 5 で事前分布を与え、Sobol 感度分析で結果への影響を調べる。
値の根拠（ドラマ内の出来事や日本の大手銀行の慣行）があるものはコメントに書く。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Params:
    start_time: float = 2013.75  # 2013年10月
    n_steps: int = 20  # 半期×20＝10年

    # --- 業績 ---
    perf_persist: float = 0.5  # 前半期の業績の持ち越し
    perf_noise: float = 0.35
    work_gain: float = 0.35  # 業務専念による業績の上昇（×能力）

    # --- 外生イベント ---
    case_fail_rate: float = 0.01  # 大型案件の失敗（半期・人物あたり）
    inspection_rate: float = 0.25  # 金融庁検査（半期あたり）
    insp_find: float = 0.03  # 検査で非公然の秘密が見つかる確率
    insp_kurosaki_bonus: float = 0.04  # 黒崎が検査を指揮する場合の上乗せ
    leak_rate: float = 0.005  # 知っている人物1人・証拠1あたりの偶発的な露見
    secret_halflife: float = 4.0  # 古い秘密ほど偶発的に露見しにくくなる（年）
    latent_scandal_rate: float = 0.015  # 派閥の未知の過去の不祥事が露見する確率（半期あたり）

    # --- 行動の判定 ---
    expose_k_evidence: float = 4.0
    expose_k_severity: float = 1.5
    expose_k_protection: float = 1.5
    expose_k_credibility: float = 1.0
    expose_fsa_bonus: float = 1.0  # 金融庁の人物による告発（検査権限）
    gather_discover: float = 0.25  # 敵の未知の秘密を見つける確率（×情報収集力）
    gather_detect: float = 0.2
    undermine_success: float = 0.5
    undermine_detect: float = 0.3
    lobby_gain: float = 0.15
    support_gain: float = 0.25
    misconduct_gain: float = 0.5

    # --- 処分 ---
    # 処分スコア s = 重大度 ×(0.5 + 0.5×証拠) + ノイズ − 寛大さ
    # s < 0.25 注意 / < 0.45 降格 / < 0.65 出向 / < 0.85 解任 / それ以上 刑事告発
    sanction_thresholds: tuple[float, float, float, float] = (0.25, 0.45, 0.65, 0.85)
    sanction_noise: float = 0.15
    # 組織防衛の重みによる寛大さ。最終回で中野渡（org_protect=0.8）が大和田（S1: 重大度0.8）を
    # 取締役への降格で留めた判断を再現する値（0.8 − 0.5×0.8 = 0.40 → 降格の範囲）
    leniency_scale: float = 0.5
    crony_leniency: float = 0.2
    toudori_resign_threshold: float = 0.35  # 頭取が当事者の場合、これ以上で引責辞任

    # --- 人事 ---
    hr_tau: float = 0.5  # 人事の選択の温度
    hr_w: dict[str, float] = field(
        default_factory=lambda: {
            "perf": 1.0,
            "ability": 0.6,
            "sponsor": 0.8,
            "relation": 1.2,
            "faction": 0.8,
            "age": 0.5,
            "scandal": 1.5,
            "reputation": 0.4,
            "young": 0.8,  # 目安の年齢より1歳若いごとの減点
        }
    )
    # 非役員の昇進（半期あたりのロジットの切片）と、昇進しにくくなる年齢
    promo_base: dict[int, float] = field(default_factory=lambda: {1: -2.0, 2: -2.4, 3: -2.8})
    promo_age_max: dict[int, float] = field(default_factory=lambda: {1: 45.0, 2: 47.0, 3: 50.0})
    return_base: float = -2.0  # 出向からの復帰
    return_max_age: float = 52.0
    # 年次による出向（日本の大手銀行では50歳前後で役員候補以外が出向・転籍する慣行）
    age_second_threshold: dict[int, float] = field(
        default_factory=lambda: {1: 50.0, 2: 50.0, 3: 50.0, 4: 53.0}
    )
    age_second_rate: float = 0.25
    transfer_age: float = 55.0
    transfer_rate: float = 0.3
    staff_retire_age: float = 60.0
    # 役員の退任年齢（ランク別）
    exec_age_limit: dict[int, float] = field(
        default_factory=lambda: {5: 60.0, 6: 60.0, 7: 62.0, 8: 63.0, 9: 64.0, 10: 72.0}
    )
    toudori_hazard: float = 0.12  # 頭取の退任（年1回の判定）。平均在任の目安6〜8年
    toudori_old_age: float = 68.0
    toudori_old_hazard: float = 0.15
    board_weight: float = 0.4  # 頭取の後継指名における取締役会の支持の重み

    # --- 金融庁 ---
    fsa_promo_base: float = -1.5
    fsa_retire_age: float = 60.0
    fsa_inspector_max_rank: int = 4  # このランクまでは検査を指揮する

    # --- 関係の変化 ---
    hostility_decay_min: float = 0.94  # 恨みが最も早く薄れる人物（grudge=0）の半期あたり保持率
    trust_decay: float = 0.98
    debt_decay: float = 0.95

    # --- NPC ---
    npc_pool_size: int = 30  # 部長級NPCの候補者数を保つ

    # --- 年次の制約 ---
    # 各ランクに就く年齢の目安（これより若いと人事スコアを減点）。
    # 常務55歳は、大和田が「歴代最速・最年少」で常務になり2013年時点で56歳というドラマ内の記述に合わせた
    rank_min_age: dict[int, float] = field(
        default_factory=lambda: {
            2: 30.0,
            3: 38.0,
            4: 44.0,
            5: 50.0,
            6: 52.0,
            7: 55.0,
            8: 57.0,
            9: 59.0,
            10: 58.0,
        }
    )
    min_tenure_staff: float = 2.5  # 非役員の昇進に必要な現ランクの在任年数
    min_tenure_exec: float = 2.0  # 役員の昇進に必要な現ランクの在任年数
    min_tenure_fsa: float = 2.0
