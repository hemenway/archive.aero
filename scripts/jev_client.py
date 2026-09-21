#!/usr/bin/env python3
"""Shared plumbing for TypeSafe's Jev (System One) API — worklist 16.

One endpoint, three question types (choice / noul / score), typed answers.
Everything project-specific (state builders, question wording, thresholds)
lives in the calling script; this module only owns:

  * the key: env TYPESAFE_API_KEY, else the repo .env (gitignored) — never
    a literal in any script;
  * the pinned model (jev-1.13.0 by default: aliases move, thresholds don't
    carry across versions) and the `model` the server actually reports;
  * retries with backoff on 429/529 (Retry-After honoured) and transient 5xx;
  * a sqlite response cache keyed on sha256(model+state+questions), so a
    re-run re-scores for free and a crash resumes where it stopped;
  * a JSONL audit log (request + response + usage) under worklists/data/jev/.

Usage:
    from jev_client import JevClient, choice, noul, score
    jev = JevClient(task="airfields_dates")
    ans = jev.ask({"narrative": text}, {"closed": choice("...", {...})})
    ans["closed"]["choice"], ans["closed"]["confidence"]
    for key, ans in jev.ask_many(items, lambda it: (it["id"], state, questions)):
        ...

Jev is text-only, does no arithmetic, and reads literally: code finds the
candidates, Jev selects among them (always with a not_stated / none option).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "worklists" / "data" / "jev"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"  # pinned on purpose (see module docstring)
USER_AGENT = "archive.aero-jev/0.1 (+https://archive.aero)"


# ---------------------------------------------------------------- questions
def choice(instructions, criteria):
    """criteria: {option: description-or-None}; answer carries choice,
    probabilities (sum to 1) and confidence."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def noul(instructions, true=None, false=None):
    """Yes/no; answer is P(yes) in `noul`; no confidence field."""
    q = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v}
    return q


def score(instructions, levels):
    """levels: ordered list of level descriptions (>= 2)."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


# ---------------------------------------------------------------- key
def load_key():
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if key:
        return key
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    sys.exit("TYPESAFE_API_KEY not set (env or repo .env)")


class JevError(RuntimeError):
    pass


# ---------------------------------------------------------------- client
class JevClient:
    def __init__(self, task, model=DEFAULT_MODEL, workers=8, cache=True, log=True,
                 timeout=90, max_retries=6):
        self.task = task
        self.model = model
        self.workers = workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.key = load_key()
        self._lock = threading.Lock()
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "cached": 0}
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._db = None
        if cache:
            self._db = sqlite3.connect(DATA_DIR / "cache.sqlite", check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, task TEXT, "
                "model TEXT, ts REAL, response TEXT)"
            )
            self._db.commit()
        self._log = open(DATA_DIR / f"{task}.jsonl", "a", encoding="utf-8") if log else None

    # -- cache -------------------------------------------------------------
    @staticmethod
    def request_key(model, state, questions):
        blob = json.dumps({"m": model, "s": state, "q": questions}, sort_keys=True,
                          ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, key):
        if self._db is None:
            return None
        with self._lock:
            row = self._db.execute("SELECT response FROM responses WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _cache_put(self, key, resp):
        if self._db is None:
            return
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)",
                (key, self.task, resp.get("model", self.model), time.time(),
                 json.dumps(resp, ensure_ascii=False)),
            )
            self._db.commit()

    # -- HTTP --------------------------------------------------------------
    def _post(self, body):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        delay = 1.0
        last = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                ENDPOINT, data=data, method="POST",
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json",
                         "User-Agent": USER_AGENT},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")[:2000]
                if e.code == 402:  # billing: no credits — never worth a retry
                    raise JevError(f"HTTP 402 (no TypeSafe API credits — top up at "
                                   f"https://console.typesafe.ai/settings/billing): {text}") from None
                if e.code in (400, 401, 403, 404, 422):
                    raise JevError(f"HTTP {e.code}: {text}") from None
                retry_after = e.headers.get("Retry-After") if e.headers else None
                last = f"HTTP {e.code}: {text[:200]}"
                wait = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else delay
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = f"{type(e).__name__}: {e}"
                wait = delay
            time.sleep(wait + random.uniform(0, 0.5))
            delay = min(delay * 2, 30)
        raise JevError(f"gave up after {self.max_retries + 1} attempts: {last}")

    # -- public ------------------------------------------------------------
    def ask(self, state, questions, tag=None):
        """Evaluate `questions` (dict id -> question) over `state`.
        Returns the answers dict (id -> answer); the raw response is cached
        and logged. `tag` is a free identifier for the audit log."""
        key = self.request_key(self.model, state, questions)
        resp = self._cache_get(key)
        cached = resp is not None
        if not cached:
            resp = self._post({"state": state, "model": self.model, "questions": questions})
            if "answers" not in resp:
                raise JevError(f"unexpected response: {json.dumps(resp)[:300]}")
            self._cache_put(key, resp)
        with self._lock:
            self.usage["requests"] += 1
            if cached:
                self.usage["cached"] += 1
            else:
                for k in ("input_tokens", "output_tokens"):
                    self.usage[k] += resp.get("usage", {}).get(k, 0)
            if self._log is not None:
                self._log.write(json.dumps({
                    "ts": time.time(), "task": self.task, "tag": tag, "key": key,
                    "cached": cached, "model": resp.get("model"), "usage": resp.get("usage"),
                    "state": state, "questions": questions, "answers": resp["answers"],
                }, ensure_ascii=False) + "\n")
                self._log.flush()
        return resp["answers"]

    def ask_many(self, items, build, progress_every=100, label=""):
        """items: iterable; build(item) -> (tag, state, questions) or None to skip.
        Yields (item, answers) in input order; errors are yielded as
        (item, JevError) so the caller decides. Runs `workers` threads."""
        items = list(items)
        results = [None] * len(items)
        done = 0
        t0 = time.time()

        def run(i):
            spec = build(items[i])
            if spec is None:
                return None
            tag, state, questions = spec
            try:
                return self.ask(state, questions, tag=tag)
            except JevError as e:
                return e

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for i, res in zip(range(len(items)), ex.map(run, range(len(items)))):
                results[i] = res
                done += 1
                if progress_every and done % progress_every == 0:
                    u = self.usage
                    print(f"  {label}{done}/{len(items)}  {time.time() - t0:.0f}s  "
                          f"in_tok={u['input_tokens']:,} cached={u['cached']}", file=sys.stderr)
        for it, res in zip(items, results):
            if res is not None:
                yield it, res

    def cost_usd(self, usd_per_mtok=0.042):
        return self.usage["input_tokens"] / 1e6 * usd_per_mtok

    def summary(self):
        u = self.usage
        return (f"{u['requests']} requests ({u['cached']} from cache), "
                f"{u['input_tokens']:,} input tokens ≈ ${self.cost_usd():.3f}, model {self.model}")


# ---------------------------------------------------------------- helpers
def top(answer, n=3):
    """[(option, p), ...] sorted by probability, for Choice/Score answers."""
    probs = answer.get("probabilities", {})
    return sorted(probs.items(), key=lambda kv: -kv[1])[:n]


if __name__ == "__main__":
    # smoke test: python3 scripts/jev_client.py
    jev = JevClient(task="smoke", cache=False, log=False)
    ans = jev.ask({"text": "Anniston Flight Service Station (ANB), Alabama, 1969"},
                  {"is_fss": noul("Does `text` name a Flight Service Station?")})
    print(ans, jev.summary())
