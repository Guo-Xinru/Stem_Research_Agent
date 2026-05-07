"""Small OpenAI wrapper for live protocol generation."""

from __future__ import annotations

import json
import os
from typing import Any, Callable


DEFAULT_MODEL = "gpt-5.5"


class LLMConfigurationError(RuntimeError):
    """Raised when live LLM mode is requested without required configuration."""


class LLMResponseError(RuntimeError):
    """Raised when the LLM response cannot be parsed or validated."""


def load_openai_config() -> dict[str, str]:
    """Load OpenAI config from environment, using python-dotenv when available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if not api_key:
        raise LLMConfigurationError(
            "OPENAI_API_KEY is required for live OpenAI modes."
        )
    return {"api_key": api_key, "model": model}


def request_strict_json(
    *,
    system_prompt: str,
    user_prompt: str,
    validate: Callable[[Any], Any],
) -> tuple[Any, dict[str, str]]:
    """Request JSON from OpenAI and validate it, retrying once on JSON/schema issues."""
    config = load_openai_config()
    client = _openai_client(config["api_key"])
    last_error: Exception | None = None

    for attempt in range(2):
        response_text = _responses_json_response(
            client=client,
            model=config["model"],
            system_prompt=system_prompt,
            user_prompt=_retry_prompt(user_prompt, last_error) if attempt else user_prompt,
        )
        try:
            parsed = json.loads(response_text)
            validated = validate(parsed)
            return validated, {"model": config["model"]}
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            last_error = exc

    raise LLMResponseError(f"OpenAI response did not produce valid protocol JSON: {last_error}")


def _openai_client(api_key: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMConfigurationError(
            "The openai package is required when --protocol-mode live is used."
        ) from exc
    return OpenAI(api_key=api_key)


def _responses_json_response(
    *,
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        text={"format": {"type": "json_object"}},
    )
    content = response.output_text
    if not content:
        raise LLMResponseError("OpenAI returned an empty protocol response.")
    return content


def _retry_prompt(user_prompt: str, error: Exception | None) -> str:
    return (
        f"{user_prompt}\n\n"
        "The previous response was not valid protocol JSON for the required schema. "
        f"Return only corrected JSON. Validation error: {error}"
    )
