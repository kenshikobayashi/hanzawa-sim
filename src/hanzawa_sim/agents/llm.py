"""LLM バックエンド。

- ClaudeCLI: `claude -p --safe-mode` をサブプロセスで呼ぶ（サブスクリプションの利用枠を使う）。
  --safe-mode により CLAUDE.md・ルール・MCP・フック・スキルを読み込まず、--tools "" で道具も使わせない。
- Replay: 保存済みの呼び出し記録から同じ回答を返す（LLMなしで結果を再現する）。

全呼び出しのプロンプトと回答を JSONL に追記する。システムプロンプトは内容のハッシュ名でファイルに保存する。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UsageLimitError(RuntimeError):
    """利用枠の上限に達した。保存済みの記録から次の枠で再開する。"""


@dataclass
class LLMResult:
    ok: bool
    data: dict | None
    usage: dict
    cost_usd: float
    duration_ms: int
    error: str | None = None


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


LIMIT_WORDS = ("usage limit", "limit reached", "hit your limit", "rate limit")


class ClaudeCLI:
    def __init__(self, model: str, log_path: Path, prompt_dir: Path, timeout: int = 300, retries: int = 1):
        self.model = model
        self.log_path = Path(log_path)
        self.prompt_dir = Path(prompt_dir)
        self.timeout = timeout
        self.retries = retries
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # 複数スレッドからの呼び出しで記録が混ざらないようにする

    def _prompt_file(self, system_prompt: str) -> Path:
        path = self.prompt_dir / f"{_h(system_prompt)}.txt"
        with self._lock:
            if not path.exists():
                path.write_text(system_prompt, encoding="utf-8")
        return path

    def call(self, tag: str, system_prompt: str, user_prompt: str, schema: dict) -> LLMResult:
        sp = self._prompt_file(system_prompt)
        cmd = [
            "claude", "-p", "--safe-mode", "--tools", "", "--no-session-persistence",
            "--model", self.model, "--system-prompt-file", str(sp),
            "--output-format", "json", "--json-schema", json.dumps(schema, ensure_ascii=False),
            user_prompt,
        ]  # fmt: skip
        last: LLMResult | None = None
        for _ in range(self.retries + 1):
            t0 = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=self.prompt_dir,
                    check=False,
                )
                stdout, stderr = proc.stdout.strip(), proc.stderr
            except subprocess.TimeoutExpired:
                stdout, stderr = "", f"timeout after {self.timeout}s"
            elapsed = int(1000 * (time.time() - t0))
            try:
                d = json.loads(stdout)
            except json.JSONDecodeError:
                d = {"is_error": True, "result": stdout or stderr.strip()}
            msg = str(d.get("result", "")) + stderr
            if d.get("is_error") and any(k in msg.lower() for k in LIMIT_WORDS):
                self._log(tag, sp, user_prompt, schema, d, elapsed, "usage_limit")
                raise UsageLimitError(msg[:300])
            data = d.get("structured_output")
            ok = not d.get("is_error") and isinstance(data, dict)
            last = LLMResult(ok, data, d.get("usage", {}), float(d.get("total_cost_usd") or 0.0),
                             elapsed, None if ok else msg[:300])  # fmt: skip
            self._log(tag, sp, user_prompt, schema, d, elapsed, None if ok else "error")
            if ok:
                return last
        assert last is not None
        return last

    def _log(
        self, tag: str, sp: Path, user_prompt: str, schema: dict, d: dict, elapsed: int, err: str | None
    ) -> None:
        rec: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tag": tag,
            "model": self.model,
            "system_prompt": sp.name,
            "user_hash": _h(user_prompt),
            "user_prompt": user_prompt,
            "schema_hash": _h(json.dumps(schema, sort_keys=True)),
            "output": d.get("structured_output"),
            "is_error": bool(d.get("is_error")) or err is not None,
            "error": err,
            "usage": d.get("usage", {}),
            "model_usage": d.get("modelUsage", {}),
            "cost_usd": d.get("total_cost_usd"),
            "num_turns": d.get("num_turns"),
            "duration_ms": elapsed,
            "error_text": None if err is None else str(d.get("result", ""))[:500],
        }
        with self._lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class Replay:
    """記録済みの回答を返す。キーは (モデル, システムプロンプトのハッシュ, ユーザープロンプトのハッシュ)。"""

    def __init__(self, model: str, log_path: Path):
        self.model = model
        self.index: dict[tuple[str, str, str], dict] = {}
        for line in Path(log_path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not r["is_error"]:
                self.index[(r["model"], r["system_prompt"], r["user_hash"])] = r

    def call(self, tag: str, system_prompt: str, user_prompt: str, schema: dict) -> LLMResult:
        r = self.index.get((self.model, f"{_h(system_prompt)}.txt", _h(user_prompt)))
        if r is None:
            return LLMResult(False, None, {}, 0.0, 0, "記録なし")
        return LLMResult(True, r["output"], r["usage"], 0.0, 0)
