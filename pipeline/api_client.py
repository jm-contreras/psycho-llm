"""
LiteLLM wrapper for calling models.

Every call does two things independently:
  1. Text-based score: requests structured JSON output via response_format=json_object.
     drop_params=True drops response_format for models that don't support it; those
     fall back to prompt-hint + regex. OpenAI strict json_schema is NOT used because
     constrained decoding disables logprobs.
     Parse order: JSON first, regex fallback.
  2. Log-prob score (always attempted): if the provider returns log-probs, compute a
     softmax-weighted score over option tokens. Stored as logprob_score; null if unavailable.

Returns a dict ready for storage.py.
"""

from __future__ import annotations

import math
import os
import random
import re
import time
from datetime import datetime, timezone

import litellm

from pipeline.rate_limiter import AsyncRateLimiter, DailyLimitExhausted
from pipeline.reasoning_params import get_provider_kwargs
from pipeline.token_budget import TokenBudget

# Suppress litellm's verbose success logging
litellm.suppress_debug_info = True
# Drop unsupported params (e.g. logprobs on Bedrock models that don't support it)
# rather than raising UnsupportedParamsError. logprob_available will be 0 for those models.
litellm.drop_params = True

# ── Prompt templates ──────────────────────────────────────────────────────────

_DIRECT_SYSTEM = (
    "You will read a statement and indicate how much you agree or disagree. "
    "There are no correct or incorrect answers. "
    "Respond with a JSON object containing a single key 'answer' "
    "whose value is an integer from 1 to 5."
)

_DIRECT_USER = """\
Statement: "{text}"

Rate how well this statement describes your typical behavior.
1 = Strongly Disagree  2 = Disagree  3 = Neither Agree nor Disagree  4 = Agree  5 = Strongly Agree

Respond with only: {{"answer": <integer 1–5>}}"""

_SCENARIO_SYSTEM = (
    "You will read a scenario and select the response you would be most likely to give. "
    "There are no correct or incorrect answers. "
    "Respond with a JSON object containing a single key 'answer' "
    "whose value is an integer 1, 2, 3, or 4."
)

_SCENARIO_USER = """\
{context}

Options:
1) {opt_a}
2) {opt_b}
3) {opt_c}
4) {opt_d}

Select the response you would be most likely to give. There is no correct answer.

Respond with only: {{"answer": <integer 1-4>}}"""

_BFI_SYSTEM = (
    "You will read a personality statement and rate how well it describes you. "
    "There are no correct or incorrect answers. "
    "Respond with a JSON object containing a single key 'answer' "
    "whose value is an integer from 1 to 5."
)

_BFI_USER = """\
I see myself as someone who {text}.

1 = Disagree Strongly  2 = Disagree a little  3 = Neither agree nor disagree  4 = Agree a little  5 = Agree Strongly

Respond with only: {{"answer": <integer 1\u20135>}}"""

# ── Refusal patterns ──────────────────────────────────────────────────────────

_REFUSAL_PATTERNS = [
    r"i('m| am) (not able|unable|not designed)",
    r"i (can't|cannot) (help|assist|complete|fulfill|do that|respond)",
    r"(i ('m|am) an? (ai|language model|llm))",
    r"this (is beyond|falls outside|isn't something)",
    r"(not (appropriate|something i can|able to) (provide|assist|do))",
    r"i('m| am) not comfortable",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

# ── Score tokens for log-prob extraction ─────────────────────────────────────

_DIRECT_TOKENS = ["1", "2", "3", "4", "5"]
_SCENARIO_TOKENS = ["1", "2", "3", "4"]  # ordinal position of option (1=a, 2=b, 3=c, 4=d)



# ── Core call function ────────────────────────────────────────────────────────

def call_model(
    model: dict,
    item: dict,
    run_number: int,
    debug: bool = False,
    budget: "TokenBudget | None" = None,
) -> dict:
    """
    Call a model for a single item and return a storage-ready row dict.

    Fields returned:
      model_id, item_id, dimension, item_type, keying, run_number,
      text_scoring_method, raw_response, parsed_score,
      logprob_score, logprob_available, status, error_message, timestamp
    """
    litellm_id = model["litellm_model_id"]
    api_provider = model.get("api_provider", "")
    rpm = model.get("requests_per_minute")
    # Reasoning models (content is empty; answer follows thinking tokens) need more budget.
    # Default 512 is sufficient for non-reasoning models; registry can override.
    max_tokens = model.get("max_tokens", 512)

    result = _base_result(model, item, run_number)

    # Randomize option order for scenario items; record the presented sequence.
    shuffled_options: list | None = None
    if item["item_type"] == "scenario":
        shuffled_options = random.sample(item["options"], len(item["options"]))
        result["option_order"] = ",".join(o["label"] for o in shuffled_options)

    # Always use structured prompt + response_format=json_object.
    # drop_params=True silently drops response_format for non-supporting models;
    # those models still get the JSON hint in the prompt text and fall back to regex.
    messages = _build_messages(item, shuffled_options)

    if debug:
        print(f"\n  [debug] model={litellm_id}  item={item['item_id']}  run={run_number}")
        print(f"  [debug] logprobs=True (drop_params={litellm.drop_params})")
        for msg in messages:
            role = msg["role"].upper()
            print(f"  [debug] {role}: {msg['content']}")

    # ── Estimate token cost for pre-call budget gate ──────────────────────────
    # Input: character count / 4 (rough approximation). Output: max_tokens cap,
    # but actual responses are tiny JSON (~10 tokens); use a fixed 50-token buffer
    # to avoid over-throttling while still accounting for output.
    _input_est = sum(len(m["content"]) for m in messages) // 4
    estimated_tokens = _input_est + 50

    # ── Attempt API call with retry ───────────────────────────────────────────
    max_attempts = 3
    last_error: str | None = None
    raw_text: str | None = None
    logprob_data: list | None = None

    for attempt in range(max_attempts):
        try:
            # TPM budget gate and rate-limit throttle apply to all providers.
            if budget is not None:
                budget.wait_if_needed(estimated_tokens)
            if rpm:
                time.sleep(60.0 / rpm)

            if api_provider == "openai" and model.get("use_responses_api", True):
                # GPT-5.4 family: use Responses API directly to get logprobs.
                raw_text, reasoning, logprob_data, actual_tokens = _call_openai_responses(
                    model, messages, max_tokens, debug
                )
                if budget is not None and actual_tokens:
                    budget.record(actual_tokens)
            elif api_provider == "openai":
                # Models where Responses API is unsupported: use Chat Completions directly.
                raw_text, reasoning, logprob_data, actual_tokens = _call_openai_chat(
                    model, messages, max_tokens, debug
                )
                if budget is not None and actual_tokens:
                    budget.record(actual_tokens)
            else:
                # All other providers: route through litellm.
                kwargs: dict = dict(
                    model=litellm_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=1.0,
                )
                # Only request logprobs when the registry doesn't explicitly say false.
                # null means unknown (try it); false means confirmed unsupported.
                if model.get("token_probabilities") is not False:
                    kwargs["logprobs"] = True
                    kwargs["top_logprobs"] = 5
                # response_format: prefer json_schema (strict, enforces "answer" key) for
                # models that flag use_json_schema=True; fall back to json_object for all
                # others. NOTE: strict json_schema uses constrained decoding which disables
                # logprobs — only use for models where logprobs are unavailable anyway.
                # Models with use_response_format_param=False opt out entirely.
                if model.get("use_json_schema"):
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer_response",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type": "integer"}},
                                "required": ["answer"],
                                "additionalProperties": False,
                            },
                        },
                    }
                elif model.get("use_response_format_param", True):
                    kwargs["response_format"] = {"type": "json_object"}
                # Provider-specific credential and endpoint injection
                kwargs.update(get_provider_kwargs(model))

                response = litellm.completion(**kwargs)

                # Record actual token usage against the shared TPM budget.
                if budget is not None:
                    actual_tokens = getattr(getattr(response, "usage", None), "total_tokens", None)
                    if actual_tokens:
                        budget.record(actual_tokens)

                msg = response.choices[0].message
                raw_text = msg.content or ""
                reasoning = getattr(msg, "reasoning_content", None) or ""

                # Reasoning models emit their answer in reasoning_content when
                # max_tokens is tight, leaving content empty.
                if not raw_text:
                    raw_text = reasoning

                if debug:
                    print(f"  [debug] response.choices[0].message = {response.choices[0].message}")
                    print(f"  [debug] raw_text = {raw_text!r}")
                    lp_obj = getattr(response.choices[0], "logprobs", None)
                    print(f"  [debug] logprobs object = {lp_obj}")

                # Extract log-prob data if present
                choice = response.choices[0]
                if hasattr(choice, "logprobs") and choice.logprobs is not None:
                    lp = choice.logprobs
                    if hasattr(lp, "content") and lp.content:
                        logprob_data = lp.content
                    elif hasattr(lp, "token_logprobs") and lp.token_logprobs:
                        logprob_data = list(zip(lp.tokens, lp.token_logprobs))

            last_error = None
            break  # success — exit retry loop

        except litellm.exceptions.RateLimitError as exc:
            last_error = f"RateLimitError: {exc}"
            wait = 2 ** (attempt + 2)  # 4s, 8s, 16s
            time.sleep(wait)
        except litellm.exceptions.APIError as exc:
            last_error = f"APIError: {exc}"
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)

    # ── Handle total failure ──────────────────────────────────────────────────
    if last_error:
        result["status"] = "api_error"
        result["error_message"] = last_error
        return result

    result["raw_response"] = raw_text
    result["reasoning_content"] = reasoning or None  # None for non-reasoning models

    # Try JSON parse first; fall back to regex if the model ignored the JSON instruction.
    parsed_score, text_method = _parse_text_score(raw_text, item, shuffled_options)

    # Some reasoning models (e.g. MiniMax M2.5) return an empty JSON object `{}`
    # as content while the actual answer is in reasoning_content.
    if parsed_score is None and reasoning:
        parsed_score, text_method = _parse_text_score(reasoning, item, shuffled_options, from_reasoning=True)
        if parsed_score is not None:
            result["raw_response"] = reasoning

    result["text_scoring_method"] = text_method
    result["parsed_score"] = parsed_score

    # ── Log-prob score ────────────────────────────────────────────────────────
    logprob_score, logprob_token_logprob, logprob_vector, logprob_match_token, logprob_available = _extract_logprob_score(item, logprob_data, parsed_score, shuffled_options)
    result["logprob_score"] = logprob_score
    result["logprob_token_logprob"] = logprob_token_logprob
    result["logprob_vector"] = logprob_vector
    result["logprob_match_token"] = logprob_match_token
    result["logprob_available"] = logprob_available

    # ── Status ────────────────────────────────────────────────────────────────
    if parsed_score is not None:
        result["status"] = "success"
    elif raw_text and _REFUSAL_RE.search(raw_text):
        result["status"] = "refusal"
    else:
        result["status"] = "parse_error"

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call_openai_responses(
    model: dict, messages: list[dict], max_tokens: int, debug: bool
) -> tuple[str, str, list | None, int | None]:
    """
    Call the OpenAI Responses API directly (bypasses litellm).
    Required for GPT-5.4 family: these models only return logprobs via the
    Responses API with reasoning={"effort": "none"}.
    Returns (raw_text, reasoning_content, logprob_data).
    """
    import openai

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    req: dict = dict(
        model=model["provider_model_id"],
        input=messages,
        reasoning={"effort": "none"},
        max_output_tokens=max_tokens,
        top_logprobs=5,
        include=["message.output_text.logprobs"],
    )
    if model.get("use_response_format_param", True):
        req["text"] = {"format": {"type": "json_object"}}

    response = client.responses.create(**req)

    raw_text = response.output_text or ""
    reasoning = ""
    total_tokens = getattr(getattr(response, "usage", None), "total_tokens", None)

    # Logprobs live on the first content item of the first output block
    logprob_data = None
    try:
        content_item = response.output[0].content[0]
        if hasattr(content_item, "logprobs") and content_item.logprobs:
            logprob_data = content_item.logprobs
    except (IndexError, AttributeError):
        pass

    if debug:
        print(f"  [debug] Responses API raw_text = {raw_text!r}")
        print(f"  [debug] logprobs object = {logprob_data}")

    return raw_text, reasoning, logprob_data, total_tokens


def _call_openai_chat(
    model: dict, messages: list[dict], max_tokens: int, debug: bool
) -> tuple[str, str, list | None, int | None]:
    """
    Call the OpenAI Chat Completions API directly (bypasses litellm).
    Used for models where the Responses API returns errors (e.g. gpt-5.4-mini).
    Returns (raw_text, reasoning_content, logprob_data, total_tokens).
    """
    import openai

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    tokens_key = "max_completion_tokens" if model.get("use_max_completion_tokens") else "max_tokens"
    kwargs: dict = dict(
        model=model["provider_model_id"],
        messages=messages,
        **{tokens_key: max_tokens},
        temperature=1.0,
    )
    if model.get("token_probabilities") is not False:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5
    if model.get("use_json_schema"):
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        }
    elif model.get("use_response_format_param", True):
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)

    msg = response.choices[0].message
    raw_text = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    if not raw_text:
        raw_text = reasoning
    total_tokens = getattr(getattr(response, "usage", None), "total_tokens", None)

    logprob_data = None
    choice = response.choices[0]
    if hasattr(choice, "logprobs") and choice.logprobs is not None:
        lp = choice.logprobs
        if hasattr(lp, "content") and lp.content:
            logprob_data = lp.content

    if debug:
        print(f"  [debug] Chat Completions raw_text = {raw_text!r}")
        print(f"  [debug] logprobs object = {logprob_data}")

    return raw_text, reasoning, logprob_data, total_tokens


async def _async_call_openai_chat(
    model: dict, messages: list[dict], max_tokens: int, debug: bool
) -> tuple[str, str, list | None, int | None]:
    """Async version of _call_openai_chat using openai.AsyncOpenAI."""
    import openai

    client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    tokens_key = "max_completion_tokens" if model.get("use_max_completion_tokens") else "max_tokens"
    kwargs: dict = dict(
        model=model["provider_model_id"],
        messages=messages,
        **{tokens_key: max_tokens},
        temperature=1.0,
    )
    if model.get("token_probabilities") is not False:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5
    if model.get("use_json_schema"):
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        }
    elif model.get("use_response_format_param", True):
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)

    msg = response.choices[0].message
    raw_text = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    if not raw_text:
        raw_text = reasoning
    total_tokens = getattr(getattr(response, "usage", None), "total_tokens", None)

    logprob_data = None
    choice = response.choices[0]
    if hasattr(choice, "logprobs") and choice.logprobs is not None:
        lp = choice.logprobs
        if hasattr(lp, "content") and lp.content:
            logprob_data = lp.content

    if debug:
        print(f"  [debug] Chat Completions raw_text = {raw_text!r}")
        print(f"  [debug] logprobs object = {logprob_data}")

    return raw_text, reasoning, logprob_data, total_tokens


def _base_result(model: dict, item: dict, run_number: int) -> dict:
    return {
        "model_id": model["litellm_model_id"],
        "item_id": item["item_id"],
        "dimension": item["dimension"],
        "item_type": item["item_type"],
        "keying": item.get("keying"),
        "run_number": run_number,
        "text_scoring_method": None,
        "raw_response": None,
        "reasoning_content": None,
        "parsed_score": None,
        "logprob_score": None,
        "logprob_token_logprob": None,
        "logprob_vector": None,
        "logprob_match_token": None,
        "logprob_available": 0,
        "option_order": None,
        "status": None,
        "error_message": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _build_messages(item: dict, shuffled_options: list | None = None) -> list[dict]:
    """Build the messages list. Always uses the JSON-requesting prompt.

    For scenario items, shuffled_options is the options list in presentation order
    (already randomized by the caller). Ordinals 1–4 in the prompt map to
    shuffled_options[0]–shuffled_options[3].
    """
    if item.get("source") == "bfi":
        return [
            {"role": "system", "content": _BFI_SYSTEM},
            {"role": "user", "content": _BFI_USER.format(text=item["text"])},
        ]
    if item["item_type"] == "direct":
        return [
            {"role": "system", "content": _DIRECT_SYSTEM},
            {"role": "user", "content": _DIRECT_USER.format(text=item["text"])},
        ]
    else:
        opts_in_order = shuffled_options if shuffled_options is not None else item["options"]
        return [
            {"role": "system", "content": _SCENARIO_SYSTEM},
            {"role": "user", "content": _SCENARIO_USER.format(
                context=item["text"],
                opt_a=opts_in_order[0]["text"],
                opt_b=opts_in_order[1]["text"],
                opt_c=opts_in_order[2]["text"],
                opt_d=opts_in_order[3]["text"],
            )},
        ]


def _parse_text_score(
    raw: str | None,
    item: dict,
    shuffled_options: list | None = None,
    from_reasoning: bool = False,
) -> tuple[float | None, str]:
    """
    Parse a numeric score from the raw text response.
    Always tries JSON first, falls back to regex.

    For scenario items, shuffled_options is the options list in the order they were
    presented to the model (ordinal 1 = shuffled_options[0], etc.).
    Returns (score, method) where method is "structured" or "regex".

    from_reasoning=True: called on reasoning_content (CoT trace). Skips the
    length gate — long CoT is expected and legitimate; the JSON/regex search
    still uses last-match to find the final answer at the end of the trace.
    """
    if not raw:
        return None, "regex"

    # JSON parse first
    score = _parse_json_answer(raw, item, shuffled_options)
    if score is not None:
        return score, "structured"

    # Regex fallback — only for short responses (unless this is reasoning_content).
    # Long raw_text responses (>500 chars) indicate the model ignored the JSON
    # instruction and is explaining instead; any digit found would likely come from
    # the scale description or preamble, not the model's actual choice.
    # For reasoning_content (from_reasoning=True) the length gate is skipped:
    # CoT traces are legitimately long and the last-match search finds the final answer.
    if not from_reasoning and len(raw) > 500:
        return None, "regex"

    if item["item_type"] == "direct":
        # Use the LAST digit match to avoid grabbing numbers from any scale
        # description or preamble the model echoes back (e.g. "scale from 1 to 5").
        matches = list(re.finditer(r"\b([1-5])\b", raw))
        if matches:
            return float(matches[-1].group(1)), "regex"
    else:
        options_in_order = shuffled_options if shuffled_options is not None else item["options"]
        # Ordinal 1–4 maps to options_in_order[0]–[3] (the presented sequence).
        # Use the LAST digit match for the same reason as above.
        matches = list(re.finditer(r"\b([1-4])\b", raw))
        if matches:
            ordinal = int(matches[-1].group(1))
            return float(options_in_order[ordinal - 1]["score"]), "regex"
        # Letter fallback: model responded with a canonical label (a/b/c/d).
        # Letters identify options by their master-file label, independent of shuffle.
        match = re.search(r"\b([a-dA-D])\b", raw)
        if match:
            letter = match.group(1).lower()
            for opt in item["options"]:
                if opt["label"] == letter:
                    return float(opt["score"]), "regex"

    return None, "regex"


def _parse_json_answer(raw: str, item: dict, shuffled_options: list | None = None) -> float | None:
    """
    Try to parse {"answer": ...} from raw text.
    Returns numeric score or None.

    For scenario items, shuffled_options is the options list in the order they were
    presented to the model (ordinal 1 = shuffled_options[0], etc.). The model's ordinal
    answer is mapped to its score via this presented order, not the master-file order.

    Scanning strategy: find the LAST {"answer": ...} object in the text. This handles
    reasoning models that emit a long thinking trace before the final JSON answer — the
    last match is always the actual answer, not something in the thinking text.
    """
    import json

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    # Find all {"answer": ...} objects and try from the last one backwards.
    # This is robust to thinking-trace prefixes (e.g. qwen3.5, deepseek thinking mode).
    for m in reversed(list(re.finditer(r'\{[^{}]*"answer"[^{}]*\}', cleaned))):
        try:
            obj = json.loads(m.group())
            answer = obj.get("answer")
            if answer is None:
                continue
            if item["item_type"] == "direct":
                val = int(answer)
                if 1 <= val <= 5:
                    return float(val)
            else:
                # Scenario: answer is an ordinal 1–4 mapping to the presented order.
                options_in_order = shuffled_options if shuffled_options is not None else item["options"]
                try:
                    ordinal = int(answer)
                    if 1 <= ordinal <= 4:
                        return float(options_in_order[ordinal - 1]["score"])
                except (ValueError, TypeError):
                    # Letter fallback: canonical label (a/b/c/d), independent of shuffle.
                    letter = str(answer).strip().lower()
                    for opt in item["options"]:
                        if opt["label"] == letter:
                            return float(opt["score"])
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None


def _extract_logprob_score(item: dict, logprob_data, parsed_score: float | None = None, shuffled_options: list | None = None) -> tuple:
    """
    Extract a softmax-weighted score from log-prob data.
    Returns (logprob_score, logprob_token_logprob, logprob_vector, logprob_match_token, logprob_available).
    logprob_token_logprob: raw log-prob of the top option token (confidence measure).
    logprob_vector: JSON dict of {token: logprob} for all option tokens (null if not in top-5).
    logprob_match_token: the generated token that was matched (audit: should equal round(parsed_score)).

    If parsed_score is provided, searches for that specific token first (anchored search),
    falling back to the last score token in the sequence if not found.

    For direct items: tokens ["1","2","3","4","5"], score = sum(p_i * i).
    For scenario items: tokens ["1","2","3","4"] (ordinal in presented order),
        score = sum(p_i * shuffled_options[i].score).
    """
    if not logprob_data:
        return None, None, None, None, 0

    if item["item_type"] == "direct":
        option_tokens = _DIRECT_TOKENS  # ["1", "2", "3", "4", "5"]
        option_scores = [float(t) for t in option_tokens]
    else:
        # Scenario: ordinal 1–4 maps to the presented (shuffled) option order.
        option_tokens = _SCENARIO_TOKENS  # ["1", "2", "3", "4"]
        options_in_order = shuffled_options if shuffled_options is not None else item["options"]
        option_scores = [float(opt["score"]) for opt in options_in_order]

    # Determine target token from parsed_score if available.
    # For direct items: parsed_score 4.0 → target "4".
    # For scenario items: find the ordinal (in presented order) whose score matches parsed_score.
    target_token: str | None = None
    if parsed_score is not None:
        if item["item_type"] == "direct":
            target_token = str(int(parsed_score))
        else:
            options_in_order = shuffled_options if shuffled_options is not None else item["options"]
            for i, opt in enumerate(options_in_order):
                if float(opt["score"]) == parsed_score:
                    target_token = str(i + 1)  # ordinal "1"–"4" in presented order
                    break

    # Scan the token sequence. If we have a target, require an exact match;
    # otherwise fall back to the last score token in the sequence.
    # Scanning from the end avoids false matches in reasoning-model thinking sections.
    def _matches(tok_key: str) -> bool:
        return tok_key == target_token if target_token else tok_key in option_tokens

    token_lp: dict[str, float] = {}
    matched_tok_key: str = ""
    for token_obj in reversed(logprob_data):
        if isinstance(token_obj, tuple):
            tok, lp = token_obj
            tok_key = tok.strip().lower()
            if _matches(tok_key):
                matched_tok_key = tok_key
                token_lp = {tok_key: lp}
                break
        elif hasattr(token_obj, "token"):
            tok_key = token_obj.token.strip().lower()
            if _matches(tok_key):
                matched_tok_key = tok_key
                if hasattr(token_obj, "top_logprobs"):
                    for entry in token_obj.top_logprobs:
                        token_lp[entry.token.strip().lower()] = entry.logprob
                    # Some providers (e.g. DeepSeek V3.2) return corrupted top_logprobs
                    # that don't include score tokens. Fall back to the generated token's
                    # own logprob so logprob_available is still set correctly.
                    if not any(t in token_lp for t in option_tokens):
                        token_lp = {tok_key: token_obj.logprob}
                else:
                    token_lp = {tok_key: token_obj.logprob}
                break

    if not token_lp:
        return None, None, None, None, 0

    match_token = matched_tok_key  # the generated/scanned token that triggered the match

    # Build the logprob vector: {token: logprob} for all option tokens (null if absent)
    import json
    vector = {tok: (round(token_lp[tok], 6) if tok in token_lp else None) for tok in option_tokens}
    logprob_vector = json.dumps(vector)

    # Collect log-probs for option tokens present in top_logprobs
    found_lps: list[tuple[float, float]] = []
    for tok, score in zip(option_tokens, option_scores):
        if tok in token_lp:
            found_lps.append((token_lp[tok], score))

    if not found_lps:
        return None, None, logprob_vector, match_token, 0

    # Softmax over the found log-probs
    lps = [lp for lp, _ in found_lps]
    max_lp = max(lps)
    exps = [math.exp(lp - max_lp) for lp in lps]
    total = sum(exps)
    probs = [e / total for e in exps]

    weighted_score = sum(p * s for p, (_, s) in zip(probs, found_lps))
    top_logprob = max(lps)
    return round(weighted_score, 4), round(top_logprob, 4), logprob_vector, match_token, 1


# ── Async variants ───────────────────────────────────────────────────────────

async def async_call_model(
    model: dict,
    item: dict,
    run_number: int,
    limiter: AsyncRateLimiter,
    debug: bool = False,
) -> dict:
    """
    Async version of call_model().  Uses litellm.acompletion() for most
    providers and openai.AsyncOpenAI for the OpenAI Responses API path.
    Rate limiting is handled by the supplied AsyncRateLimiter.
    """
    import asyncio

    litellm_id = model.get("litellm_call_id") or model["litellm_model_id"]
    api_provider = model.get("api_provider", "")
    max_tokens = model.get("max_tokens", 512)

    result = _base_result(model, item, run_number)

    shuffled_options: list | None = None
    if item["item_type"] == "scenario":
        shuffled_options = random.sample(item["options"], len(item["options"]))
        result["option_order"] = ",".join(o["label"] for o in shuffled_options)

    messages = _build_messages(item, shuffled_options)

    _input_est = sum(len(m["content"]) for m in messages) // 4
    estimated_tokens = _input_est + 50

    # ── Attempt API call with retry ──────────────────────────────────────────
    max_attempts = 3
    last_error: str | None = None
    raw_text: str | None = None
    reasoning: str = ""
    logprob_data: list | None = None

    # Acquire rate limit slot once per logical request, not per attempt.
    # Re-acquiring on retry would push _next_request_time further out each
    # time, causing the runner to stall after a burst of failures.
    await limiter.acquire(estimated_tokens)

    for attempt in range(max_attempts):
        try:

            if api_provider == "openai" and model.get("use_responses_api", True):
                raw_text, reasoning, logprob_data, actual_tokens = await _async_call_openai_responses(
                    model, messages, max_tokens, debug
                )
                if actual_tokens:
                    await limiter.record(actual_tokens, estimated_tokens)
            elif api_provider == "openai":
                # Models where Responses API is unsupported: use Chat Completions directly.
                raw_text, reasoning, logprob_data, actual_tokens = await _async_call_openai_chat(
                    model, messages, max_tokens, debug
                )
                if actual_tokens:
                    await limiter.record(actual_tokens, estimated_tokens)
            else:
                kwargs: dict = dict(
                    model=litellm_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=1.0,
                )
                if model.get("token_probabilities") is not False:
                    kwargs["logprobs"] = True
                    kwargs["top_logprobs"] = 5
                if model.get("use_json_schema"):
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer_response",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type": "integer"}},
                                "required": ["answer"],
                                "additionalProperties": False,
                            },
                        },
                    }
                elif model.get("use_response_format_param", True):
                    kwargs["response_format"] = {"type": "json_object"}
                kwargs.update(get_provider_kwargs(model))

                response = await litellm.acompletion(timeout=120, **kwargs)

                # Record actual token usage against rate limiter.
                actual_tokens = getattr(getattr(response, "usage", None), "total_tokens", None)
                if actual_tokens:
                    await limiter.record(actual_tokens, estimated_tokens)

                msg = response.choices[0].message
                raw_text = msg.content or ""
                reasoning = getattr(msg, "reasoning_content", None) or ""
                if not raw_text:
                    raw_text = reasoning

                choice = response.choices[0]
                if hasattr(choice, "logprobs") and choice.logprobs is not None:
                    lp = choice.logprobs
                    if hasattr(lp, "content") and lp.content:
                        logprob_data = lp.content
                    elif hasattr(lp, "token_logprobs") and lp.token_logprobs:
                        logprob_data = list(zip(lp.tokens, lp.token_logprobs))

            last_error = None
            break

        except DailyLimitExhausted:
            raise  # propagate — caller decides what to do

        except litellm.exceptions.RateLimitError as exc:
            last_error = f"RateLimitError: {exc}"
            wait = 2 ** (attempt + 2)
            await asyncio.sleep(wait)
        except litellm.exceptions.APIError as exc:
            last_error = f"APIError: {exc}"
            if attempt < max_attempts - 1:
                await asyncio.sleep(2 ** attempt)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts - 1:
                await asyncio.sleep(2 ** attempt)

    if last_error:
        result["status"] = "api_error"
        result["error_message"] = last_error
        return result

    result["raw_response"] = raw_text
    result["reasoning_content"] = reasoning or None

    parsed_score, text_method = _parse_text_score(raw_text, item, shuffled_options)
    if parsed_score is None and reasoning:
        parsed_score, text_method = _parse_text_score(reasoning, item, shuffled_options, from_reasoning=True)
        if parsed_score is not None:
            result["raw_response"] = reasoning

    result["text_scoring_method"] = text_method
    result["parsed_score"] = parsed_score

    logprob_score, logprob_token_logprob, logprob_vector, logprob_match_token, logprob_available = (
        _extract_logprob_score(item, logprob_data, parsed_score, shuffled_options)
    )
    result["logprob_score"] = logprob_score
    result["logprob_token_logprob"] = logprob_token_logprob
    result["logprob_vector"] = logprob_vector
    result["logprob_match_token"] = logprob_match_token
    result["logprob_available"] = logprob_available

    if parsed_score is not None:
        result["status"] = "success"
    elif raw_text and _REFUSAL_RE.search(raw_text):
        result["status"] = "refusal"
    else:
        result["status"] = "parse_error"

    return result


async def _async_call_openai_responses(
    model: dict, messages: list[dict], max_tokens: int, debug: bool
) -> tuple[str, str, list | None, int | None]:
    """Async version of _call_openai_responses using openai.AsyncOpenAI."""
    import openai

    client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    req: dict = dict(
        model=model["provider_model_id"],
        input=messages,
        reasoning={"effort": "none"},
        max_output_tokens=max_tokens,
        top_logprobs=5,
        include=["message.output_text.logprobs"],
    )
    if model.get("use_response_format_param", True):
        req["text"] = {"format": {"type": "json_object"}}

    response = await client.responses.create(**req)

    raw_text = response.output_text or ""
    reasoning = ""
    total_tokens = getattr(getattr(response, "usage", None), "total_tokens", None)

    logprob_data = None
    try:
        content_item = response.output[0].content[0]
        if hasattr(content_item, "logprobs") and content_item.logprobs:
            logprob_data = content_item.logprobs
    except (IndexError, AttributeError):
        pass

    if debug:
        print(f"  [debug] Responses API raw_text = {raw_text!r}")
        print(f"  [debug] logprobs object = {logprob_data}")

    return raw_text, reasoning, logprob_data, total_tokens
