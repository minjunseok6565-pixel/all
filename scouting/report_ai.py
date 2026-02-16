from __future__ import annotations

"""LLM writer for scouting reports.

This module is intentionally isolated so the DB/service logic can be tested
without requiring a network call. In production, report generation happens at
month-end checkpoint (Option A).

Implementation uses Google Generative AI (Gemini), matching existing project
modules (news_ai.py, season_report_ai.py).
"""

import datetime as _dt
import json
from typing import Any, Dict, Tuple

import google.generativeai as genai


DEFAULT_MODEL_NAME = "gemini-3-pro-preview"
PROMPT_VERSION = "scouting_report_v1"


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _extract_text_from_gemini_response(resp: Any) -> str:
    """Robustly extract plain text from gemini response."""
    # Common shape: response.text
    try:
        t = getattr(resp, "text", None)
        if isinstance(t, str) and t.strip():
            return t
    except Exception:
        pass

    # Fallback: iterate candidates/parts
    try:
        cands = getattr(resp, "candidates", None)
        if isinstance(cands, list) and cands:
            parts = getattr(cands[0].content, "parts", None)
            if isinstance(parts, list) and parts:
                texts = []
                for p in parts:
                    tx = getattr(p, "text", None)
                    if isinstance(tx, str) and tx:
                        texts.append(tx)
                if texts:
                    return "\n".join(texts)
    except Exception:
        pass

    # Ultimate fallback
    return str(resp) if resp is not None else ""


def _build_prompt(payload: Dict[str, Any]) -> str:
    """Return a Korean scouting report prompt with the structured payload."""
    # Keep JSON fairly compact to reduce tokens.
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    return (
        "당신은 NBA 구단 스카우터다. 아래의 구조화된 데이터(payload)를 바탕으로 '스카우팅 리포트'를 작성하라.\n"
        "- 출력 언어: 한국어\n"
        "- 형식: Markdown 텍스트(제목/소제목/불릿)\n"
        "- 길이: 600~1200자 내외(너무 길게 쓰지 말 것)\n"
        "- 데이터에 없는 사실을 단정하지 말고, 불확실성은 '추가 관찰 필요'로 표현하라.\n"
        "- 숫자(0~100 추정치, sigma)를 그대로 나열하지 말고, grade(A+~F), confidence(high/medium/low), 범위(range_2sigma)를 자연어로 풀어라.\n"
        "- 보고서에는 다음 섹션을 반드시 포함하라:\n"
        "  1) 요약(2~3문장)\n"
        "  2) 강점(3~5개 불릿)\n"
        "  3) 약점/리스크(2~4개 불릿)\n"
        "  4) NBA 역할/성장 시나리오(1~2문단)\n"
        "  5) 다음 체크 포인트(추가로 보면 좋을 것 2~3개)\n"
        "\n"
        "[payload JSON]\n"
        f"{payload_json}\n"
    )


class ScoutingReportWriter:
    """Thin wrapper around Gemini model to generate report_text."""

    def __init__(self, *, api_key: str, model_name: str = DEFAULT_MODEL_NAME):
        if not api_key or not str(api_key).strip():
            raise ValueError("api_key is required")
        self.api_key = str(api_key).strip()
        self.model_name = str(model_name).strip() or DEFAULT_MODEL_NAME

        # Configure globally for this process. This matches other modules.
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def write(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        prompt = _build_prompt(payload)

        # Note: We intentionally keep generation config default for now.
        # If you need more deterministic output, set temperature=0.2 etc.
        resp = self.model.generate_content(prompt)
        text = _extract_text_from_gemini_response(resp).strip()

        meta: Dict[str, Any] = {
            "model": self.model_name,
            "prompt_version": PROMPT_VERSION,
            "generated_at": _now_iso(),
        }

        # Try to capture usage metadata if available
        try:
            usage = getattr(resp, "usage_metadata", None)
            if usage is not None:
                # Convert to JSON-serializable dict-ish
                meta["usage_metadata"] = json.loads(json.dumps(usage, default=str))
        except Exception:
            pass

        return text, meta
