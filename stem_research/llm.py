"""Small OpenAI wrapper for live protocol generation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_MODEL = DEFAULT_OPENAI_MODEL


class LLMConfigurationError(RuntimeError):
    """Raised when live LLM mode is requested without required configuration."""


class LLMResponseError(RuntimeError):
    """Raised when the LLM response cannot be parsed or validated."""


def load_openai_config() -> dict[str, str]:
    """Load OpenAI config from .env and environment variables."""
    _load_env_file()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
    if not api_key:
        error = LLMConfigurationError("OPENAI_API_KEY is required for live OpenAI modes.")
        _print_openai_diagnostic(
            step="OpenAI configuration",
            model=model,
            api_key_detected=False,
            exc=error,
        )
        raise error
    return {"api_key": api_key, "model": model}


def _load_env_file() -> None:
    """Load .env without making python-dotenv mandatory for offline runs."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_file_without_dotenv(Path(".env"))
        return

    load_dotenv(dotenv_path=Path(".env"))


def _load_env_file_without_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def request_strict_json(
    *,
    system_prompt: str,
    user_prompt: str,
    validate: Callable[[Any], Any],
) -> tuple[Any, dict[str, str]]:
    """Request JSON from OpenAI and validate it, retrying once on malformed JSON."""
    config = load_openai_config()
    model = config["model"]
    api_key_detected = bool(config["api_key"])
    client = _openai_client(config["api_key"], model=model, api_key_detected=api_key_detected)
    last_json_error: json.JSONDecodeError | None = None

    for attempt in range(2):
        response_text = _responses_json_response(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=_retry_prompt(user_prompt, last_json_error) if attempt else user_prompt,
            api_key_detected=api_key_detected,
        )
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            last_json_error = exc
            if attempt == 0:
                continue
            error = LLMResponseError(f"OpenAI response was not valid JSON: {exc}")
            _print_openai_diagnostic(
                step="JSON parsing",
                model=model,
                api_key_detected=api_key_detected,
                exc=error,
            )
            raise error from exc

        try:
            validated = validate(parsed)
        except (ValueError, TypeError) as exc:
            error = LLMResponseError(f"OpenAI response failed protocol validation: {exc}")
            _print_openai_diagnostic(
                step="protocol validation",
                model=model,
                api_key_detected=api_key_detected,
                exc=error,
            )
            raise error from exc
        return validated, {"model": model}

    raise LLMResponseError(f"OpenAI response did not produce valid JSON: {last_json_error}")


def _openai_client(api_key: str, *, model: str, api_key_detected: bool):
    try:
        from openai import OpenAI
    except ImportError as exc:
        error = LLMConfigurationError(
            "The openai package is required when --protocol-mode live is used."
        )
        _print_openai_diagnostic(
            step="OpenAI import",
            model=model,
            api_key_detected=api_key_detected,
            exc=error,
        )
        raise error from exc
    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:
        _print_openai_diagnostic(
            step="OpenAI client creation",
            model=model,
            api_key_detected=api_key_detected,
            exc=exc,
        )
        raise


def _responses_json_response(
    *,
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key_detected: bool,
) -> str:
    try:
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            text={"format": {"type": "json_object"}},
        )
    except TypeError:
        response = _plain_responses_json_request(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key_detected=api_key_detected,
        )
    except Exception as exc:
        _print_openai_diagnostic(
            step="OpenAI API request",
            model=model,
            api_key_detected=api_key_detected,
            exc=exc,
        )
        raise

    content = _extract_response_text(response)
    if not content:
        error = LLMResponseError("OpenAI returned an empty protocol response.")
        _print_openai_diagnostic(
            step="OpenAI API response",
            model=model,
            api_key_detected=api_key_detected,
            exc=error,
        )
        raise error
    return content


def _plain_responses_json_request(
    *,
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key_detected: bool,
):
    try:
        return client.responses.create(
            model=model,
            instructions=system_prompt,
            input=(
                f"{user_prompt}\n\n"
                "Return only valid JSON. Do not include markdown fences or prose."
            ),
        )
    except Exception as exc:
        _print_openai_diagnostic(
            step="OpenAI API request",
            model=model,
            api_key_detected=api_key_detected,
            exc=exc,
        )
        raise


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()
    return str(response).strip()


def _retry_prompt(user_prompt: str, error: Exception | None) -> str:
    return (
        f"{user_prompt}\n\n"
        "The previous response was not valid protocol JSON for the required schema. "
        f"Return only corrected JSON. Validation error: {error}"
    )


def _print_openai_diagnostic(
    *,
    step: str,
    model: str,
    api_key_detected: bool,
    exc: BaseException,
) -> None:
    print(
        "OpenAI diagnostic: "
        f"step={step}; "
        f"model={model}; "
        f"OPENAI_API_KEY detected={'yes' if api_key_detected else 'no'}; "
        f"exception={exc.__class__.__name__}; "
        f"message={exc}",
        file=sys.stderr,
    )
