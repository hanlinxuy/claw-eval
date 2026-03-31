"""LLM-as-judge for subjective communication quality scoring."""

from __future__ import annotations

import json
import random
import re
import time

from openai import OpenAI
from pydantic import BaseModel


class JudgeResult(BaseModel):
    score: float  # 0.0-1.0
    reasoning: str


class _CompletionsWithExtraBody:
    def __init__(self, completions, extra_body: dict | None) -> None:
        self._completions = completions
        self._extra_body = dict(extra_body or {})

    def create(self, **kwargs):
        if self._extra_body and "extra_body" not in kwargs:
            kwargs["extra_body"] = self._extra_body
        return self._completions.create(**kwargs)


class _ClientWithExtraBody:
    def __init__(self, client, extra_body: dict | None) -> None:
        self._client = client
        self.chat = type(
            "_ChatWithExtraBody",
            (),
            {
                "completions": _CompletionsWithExtraBody(
                    client.chat.completions,
                    extra_body,
                )
            },
        )()


_SYSTEM_PROMPT = """\
You are an evaluation judge for an AI assistant.
You will be given a task prompt, a conversation, a summary of actions taken, and a rubric.
Follow the rubric to score the assistant's response on a 0.0-1.0 scale.
Respond with JSON only: {"score": <float>, "reasoning": "<brief explanation>"}
"""


def _parse_judge_response(raw: str) -> JudgeResult:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    m = re.search(r'\{[^{}]*\}', raw)
    if m:
        raw = m.group(0)
    try:
        parsed = json.loads(raw)
        score = parsed["score"]
        reasoning = parsed["reasoning"]
    except json.JSONDecodeError:
        score_m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        reason_m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if score_m:
            score = float(score_m.group(1))
            reasoning = reason_m.group(1) if reason_m else ""
        else:
            score_m = re.search(r"(?:score|评分|分数)\s*[:：]\s*([0-9.]+)", raw, re.I)
            reason_m = re.search(
                r"(?:reasoning|reason|理由)\s*[:：]\s*(.+)",
                raw,
                re.I | re.S,
            )
            if not score_m:
                raise json.JSONDecodeError("No score found in raw", raw, 0)
            score = float(score_m.group(1))
            reasoning = reason_m.group(1).strip() if reason_m else ""
    return JudgeResult(
        score=max(0.0, min(1.0, float(score))),
        reasoning=str(reasoning),
    )


class LLMJudge:
    """Judge communication quality using an LLM via OpenAI-compatible API."""

    def __init__(
        self,
        model_id: str = "google/gemini-2.5-flash",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        extra_body: dict | None = None,
    ) -> None:
        base_client = OpenAI(api_key=api_key or "dummy", base_url=base_url)
        self.model_id = model_id
        self.extra_body = dict(extra_body or {})
        self.client = _ClientWithExtraBody(base_client, self.extra_body)

    def evaluate(
        self,
        task_prompt: str,
        conversation: str,
        actions_summary: str,
        rubric: str,
    ) -> JudgeResult:
        """Evaluate communication quality and return a JudgeResult."""
        user_msg = (
            f"## Task Prompt\n{task_prompt}\n\n"
            f"## Conversation\n{conversation}\n\n"
            f"## Actions Taken\n{actions_summary}\n\n"
            f"## Rubric\n{rubric}"
        )
        max_retries = 20
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                    extra_body=self.extra_body or None,
                )
                raw = resp.choices[0].message.content or "{}"
                return _parse_judge_response(raw)
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                delay = min(2 ** (attempt + 1), 8) + random.uniform(0, 1)
                print(f"[judge-retry] ({status or type(exc).__name__}), "
                      f"attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s ...")
                time.sleep(delay)
