# hanzawa-sim

ドラマ「半沢直樹」（2013年版）の最終回直後の東京中央銀行を初期状態とし、登場人物15人をそれぞれ独立した LLM エージェントにして、その後10年間の人事と権力闘争を10万回シミュレーションするプロジェクトです。

- 登場人物ごとに、性格・価値観・目標（ペルソナ）と、その人物が最終回の時点で知っている事実だけを持つエージェントを作ります
- 10万回すべてを LLM で動かすことはできないため、LLM エージェントに状況カードを見せて各行動の確率を答えさせ、その回答から人物ごとの行動方策（多項ロジット）を学習し（蒸留）、学習した方策で10万回を実行します
- 昇進・降格は、人事スコアの要因分解、イベントの原因リンクの追跡、介入実験（共通乱数による対の比較）で分析します

## 解説ページ

| ページ | 内容 |
|---|---|
| [docs/method.md](docs/method.md) | シミュレーションの仕組み（初期状態、LLMエージェント、状況カード、方策の学習、ゲームマスターの規則、分析と検証の方法） |
| [docs/results.md](docs/results.md) | 結果の解説（10年後の役職、人物ごとの結果、頭取の後継、対立の構造、介入実験、感度分析、モデルによる違い） |
| [PLAN.md](PLAN.md) | 計画と、計画から変えた点の記録 |
| [data/results/sonnet/report.md](data/results/sonnet/report.md) | 集計の表（自動生成） |

> 本プロジェクトはファンによる非公式の分析です。ペルソナと事実の整理は作品を参考に独自の言葉で書いたもので、台詞や画像は含みません。

## 主な結果（Sonnet の方策、10万回）

| 人物 | 10年以内に役員 | 10年後: 部長 | 出向 | 失脚 |
|---|---|---|---|---|
| 半沢直樹 | 24.1% | 32.6% | 17.3% | 5.3% |
| 渡真利忍 | 35.1% | 40.3% | 11.9% | 0.6% |
| 近藤直弼 | 19.7% | 44.1% | 22.6% | 0.4% |

- 半沢が10年以内に頭取になる試行はなかった。中野渡頭取の後継は、ほぼすべてドラマに登場しない人物（旧東京第一銀行出身 54%、旧産業中央銀行出身 45%）
- 半沢の後退の最大の引き金は黒崎の告発（56.6% の試行）で、処分の元になった秘密の大半は半沢自身の疎開資料の件
- 介入実験: 疎開資料の件が最初に公になると半沢の役員到達率は −9.2 ポイント、中野渡が直後に退任すると −5.2 ポイント。大和田が直後に退職しても半沢は変わらず（+0.9、有意でない）、黒崎が −10.9 ポイント
- 感度分析の9パラメータを 0.5〜1.5倍に変えても、半沢の役員到達率の変化は ±2 ポイント以内。一方、Haiku の回答から学習した方策では 12.6% になり、LLM のモデルの選び方の影響が最も大きい
- 方策モデルと LLM の回答の差（検証用カード、全変動距離）は 0.078。LLM 自体の再回答の差は 0.047

詳しくは [docs/results.md](docs/results.md) を参照してください。

## 構成

| パス | 内容 |
|---|---|
| `data/world/` | 人物15人の初期状態、秘密（不正）10件、人間関係47本 |
| `data/personas/personas.yaml` | 15人のペルソナ |
| `src/hanzawa_sim/world/` | 世界モデル（人物・秘密・関係・パラメータ） |
| `src/hanzawa_sim/sim/` | ゲームマスター（行動の解決、人事、処分、外生イベント）、乱数表、並列実行 |
| `src/hanzawa_sim/agents/` | システムプロンプト、状況カード、`claude -p` の呼び出し、方策の学習、完全LLMラン |
| `src/hanzawa_sim/analysis/` | イベントの集計（後退の引き金、昇進の要因、対立の構造） |
| `scripts/pilot.py` | パイロット（18回の呼び出し） |
| `scripts/elicit.py` | 方策抽出の本番（Sonnet・Haiku 各約1,150回の呼び出し。中断しても再開できる） |
| `scripts/fit_policies.py` | 方策モデルの学習と誤差の評価 |
| `scripts/analyze.py` | 10万回のモンテカルロ、介入実験、感度分析 |
| `scripts/full_llm_run.py` | 完全LLMラン（全員の行動を毎半期 LLM が決める1試行） |
| `scripts/report.py` | 結果の表（Markdown） |

## 再現手順

Python 3.12 と [uv](https://docs.astral.sh/uv/) を使います。

```bash
uv sync
uv run pytest                                   # テスト
uv run python scripts/fit_policies.py           # 保存済みの LLM の回答から方策を学習（LLM 呼び出しなし）
uv run python scripts/analyze.py baseline --policy sonnet --runs 100000
uv run python scripts/analyze.py interventions --policy sonnet --runs 10000
uv run python scripts/report.py --policy sonnet --runs 100000 --iv-runs 10000 --sens-runs 5000
```

LLM の回答はすべて `data/elicitation/` に保存してあるため、学習以降は LLM なしで再現できます。LLM の呼び出しからやり直す場合は、Claude Code（`claude` コマンド）にログインした状態で `scripts/elicit.py run` を実行します。呼び出しは `claude -p --safe-mode --tools "" --no-session-persistence` で行い、ユーザー設定やツールを読み込みません。
