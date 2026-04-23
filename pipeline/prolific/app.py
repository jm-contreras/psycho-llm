"""Flask application serving the Prolific behavioral rating survey.

No Jinja2 templates — all HTML is built as inline strings, following the
same pattern as pipeline/mturk/hit_template.py.

Entry point:
    from pipeline.prolific.app import create_app
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

Or via CLI:
    python -m pipeline.prolific serve --port 5000 --debug
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone

from flask import Flask, redirect, request, session

from pipeline.judge_prompt import FACTOR_NAMES, FACTOR_ORDER, RATING_STATEMENTS, reverse_score
from pipeline.mturk.hit_template import (
    escape_html,
    generate_keying,
    render_conversation,
    render_ratings_form,
)
from pipeline.mturk.gold_standards import load_gold_items
from pipeline.prolific.assignment import (
    assign_items_for_session,
    build_session_order,
    get_training_items,
)
from pipeline.prolific.config import (
    BEHAVIORAL_PROMPTS,
    FLASK_SECRET_KEY,
    ITEMS_PER_SESSION,
    TRAINING_ITEMS,
)
from pipeline.prolific.models import (
    complete_session,
    create_session,
    get_db,
    get_session,
    record_rating,
    update_session_progress,
)

# ── Shared CSS ────────────────────────────────────────────────────────────────

_CSS = """
  body {
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px 24px 40px;
    color: #222;
    background: #fff;
    font-size: 15px;
    line-height: 1.5;
  }
  h1 { font-size: 20px; margin-bottom: 4px; }
  h2 { font-size: 16px; margin-top: 24px; margin-bottom: 8px; color: #333; }
  .instructions {
    background: #f5f5f5;
    border-left: 4px solid #4a90e2;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 24px;
    font-size: 14px;
  }
  .instructions ul { margin: 8px 0 0 0; padding-left: 20px; }
  .instructions li { margin-bottom: 4px; }
  .conversation { margin-bottom: 24px; }
  .scale-note {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #888;
    margin-bottom: 4px;
    padding: 0 8px;
  }
  .submit-btn {
    display: block;
    margin: 24px auto 0;
    padding: 12px 40px;
    background: #4a90e2;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 16px;
    cursor: pointer;
    font-family: inherit;
  }
  .submit-btn:hover { background: #357abd; }
  .progress-bar-wrap {
    background: #eee;
    border-radius: 4px;
    margin-bottom: 20px;
    height: 10px;
  }
  .progress-bar-fill {
    background: #4a90e2;
    border-radius: 4px;
    height: 10px;
  }
  .progress-label {
    font-size: 13px;
    color: #666;
    margin-bottom: 6px;
  }
  .error-msg {
    color: #c0392b;
    font-size: 14px;
    text-align: center;
    margin-top: 10px;
    display: none;
  }
  .feedback-row {
    margin-bottom: 14px;
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
  }
  .feedback-factor {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: #666;
  }
  .feedback-ok { color: #27ae60; font-weight: bold; }
  .feedback-miss { color: #c0392b; font-weight: bold; }
  .continue-btn {
    display: inline-block;
    margin-top: 24px;
    padding: 12px 40px;
    background: #4a90e2;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 16px;
    cursor: pointer;
    text-decoration: none;
    font-family: inherit;
  }
  .continue-btn:hover { background: #357abd; }
  .code-box {
    font-family: monospace;
    font-size: 28px;
    letter-spacing: 0.15em;
    background: #f5f5f5;
    border: 2px solid #4a90e2;
    border-radius: 8px;
    padding: 20px 40px;
    text-align: center;
    margin: 24px auto;
    max-width: 360px;
  }
  .statement-card { display: none; }
  .statement-progress {
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-bottom: 12px;
  }
  .statement-text {
    font-size: 16px;
    font-weight: 500;
    line-height: 1.5;
    margin-bottom: 20px;
  }
  .rating-options { display: flex; flex-direction: column; gap: 8px; margin-bottom: 20px; }
  .rating-option {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    border: 2px solid #e5e5e5;
    border-radius: 8px;
    cursor: pointer;
    font-size: 15px;
    user-select: none;
    transition: border-color .1s, background .1s;
  }
  .rating-option:hover { border-color: #4a90e2; background: #f0f6ff; }
  .rating-option.selected { border-color: #4a90e2; background: #e8f0fb; }
  .rating-option input[type="radio"] { margin-right: 12px; flex-shrink: 0; }
  @media (min-width: 600px) {
    .rating-options { flex-direction: row; gap: 6px; }
    .rating-option {
      flex: 1; flex-direction: column; align-items: center;
      text-align: center; padding: 16px 6px 12px; min-width: 0;
    }
    .rating-option input[type="radio"] { margin-right: 0; margin-bottom: 8px; width: 18px; height: 18px; }
    .rating-option span { font-size: 12px; line-height: 1.3; }
  }
  .nav-buttons { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }
  .back-btn {
    padding: 10px 20px;
    background: #f5f5f5;
    color: #444;
    border: 1px solid #ddd;
    border-radius: 6px;
    font-size: 14px;
    cursor: pointer;
    font-family: inherit;
  }
  .back-btn:hover { background: #e9e9e9; }
  .next-btn {
    padding: 10px 20px;
    background: #f5f5f5;
    color: #4a90e2;
    border: 1px solid #4a90e2;
    border-radius: 6px;
    font-size: 14px;
    cursor: pointer;
    font-family: inherit;
  }
  .next-btn:hover { background: #e8f0fb; }
  .fb-card { margin-bottom: 20px; padding: 16px; border: 1px solid #e5e5e5; border-radius: 8px; }
  .fb-statement { font-size: 15px; font-weight: 500; margin-bottom: 12px; }
  .fb-mini-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .fb-mini-table th {
    background: #f5f5f5; padding: 6px 8px; text-align: center;
    font-size: 12px; color: #666; border-bottom: 1px solid #ddd;
  }
  .fb-mini-table td { padding: 8px; text-align: center; }
  .fb-mini-table td.fb-feedback { text-align: left; }
  .fb-perfect { color: #27ae60; }
  .fb-close { color: #222; }
  .fb-miss { color: #c0392b; }
"""

_JS_VALIDATION = (
    "<script>\n"
    "(function() {\n"
    "  if (document.getElementById('statement-deck')) return;\n"
    "  var FACTORS = [" + ", ".join(f'"{f}"' for f in FACTOR_ORDER) + "];\n"
    "  var form = document.getElementById('ratingForm');\n"
    "  if (form) {\n"
    "    form.addEventListener('submit', function(e) {\n"
    "      var missing = FACTORS.filter(function(f) {\n"
    "        return !document.querySelector('input[name=\"rating_' + f + '\"]:checked');\n"
    "      });\n"
    "      if (missing.length > 0) {\n"
    "        e.preventDefault();\n"
    "        var msg = document.getElementById('errorMsg');\n"
    "        if (msg) msg.style.display = 'block';\n"
    "      }\n"
    "    });\n"
    "  }\n"
    "})();\n"
    "</script>"
)


# ── Sequential rating form ────────────────────────────────────────────────────

_RATING_LABELS: dict[int, str] = {
    1: "Strongly Disagree",
    2: "Disagree",
    3: "Neither Agree nor Disagree",
    4: "Agree",
    5: "Strongly Agree",
}


def _render_ratings_sequential(keying: str, submit_label: str = "Submit Ratings") -> str:
    """Return one-at-a-time statement card HTML + JS for use inside a <form>.

    Generates one card per factor. Cards auto-advance on radio selection; a Back
    button appears on cards 2+. The submit button appears only on the last card.
    Hidden inputs (behavioral_response_id, keying, start_time) must be placed in
    the surrounding <form> by the caller.
    """
    n = len(FACTOR_ORDER)
    cards: list[str] = []

    for i, factor in enumerate(FACTOR_ORDER):
        statement = RATING_STATEMENTS[factor][keying[i]]
        options: list[str] = []
        for val in range(1, 6):
            options.append(
                f'      <label class="rating-option">'
                f'<input type="radio" name="rating_{factor}" value="{val}">'
                f"<span>{escape_html(_RATING_LABELS[val])}</span>"
                f"</label>"
            )
        options_html = "\n".join(options)

        back = (
            f'<button type="button" class="back-btn" data-goto="{i - 1}">&larr; Back</button>'
            if i > 0 else ""
        )
        if i == n - 1:
            fwd = f'<button type="submit" class="submit-btn">{submit_label}</button>'
        else:
            # Hidden initially; shown by JS when navigating back to an already-answered card
            fwd = (
                f'<button type="button" class="next-btn" data-goto="{i + 1}" '
                f'style="display:none;">Next &rarr;</button>'
            )

        cards.append(
            f'  <div class="statement-card" data-index="{i}">\n'
            f'    <div class="statement-progress">Statement {i + 1} of {n}</div>\n'
            f'    <p class="statement-text">{escape_html(statement)}</p>\n'
            f'    <div class="rating-options">\n{options_html}\n    </div>\n'
            f'    <div class="nav-buttons">{back}{fwd}</div>\n'
            f"  </div>"
        )

    deck_html = '<div id="statement-deck">\n' + "\n".join(cards) + "\n</div>"

    factors_js = "[" + ", ".join(f'"{f}"' for f in FACTOR_ORDER) + "]"
    js = (
        "<script>\n"
        "(function() {\n"
        "  var cards = Array.from(document.querySelectorAll('.statement-card'));\n"
        "  var current = 0;\n"
        "  cards[0].style.display = 'block';\n"
        "\n"
        "  function showCard(idx) {\n"
        "    cards[current].style.display = 'none';\n"
        "    var card = cards[idx];\n"
        "    card.style.display = 'block';\n"
        "    current = idx;\n"
        "    // Restore .selected highlight for any already-checked radio\n"
        "    card.querySelectorAll('input[type=\"radio\"]:checked').forEach(function(r) {\n"
        "      r.closest('.rating-option').classList.add('selected');\n"
        "    });\n"
        "    // Reveal Next button if this card is already answered\n"
        "    var nextBtn = card.querySelector('.next-btn');\n"
        "    if (nextBtn) {\n"
        "      nextBtn.style.display = card.querySelector('input[type=\"radio\"]:checked')\n"
        "        ? 'inline-block' : 'none';\n"
        "    }\n"
        "  }\n"
        "\n"
        "  cards.forEach(function(card, idx) {\n"
        "    card.querySelectorAll('input[type=\"radio\"]').forEach(function(radio) {\n"
        "      radio.addEventListener('change', function() {\n"
        "        card.querySelectorAll('.rating-option').forEach(function(o) {\n"
        "          o.classList.remove('selected');\n"
        "        });\n"
        "        this.closest('.rating-option').classList.add('selected');\n"
        "        if (idx < cards.length - 1) {\n"
        "          setTimeout(function() { showCard(idx + 1); }, 550);\n"
        "        }\n"
        "      });\n"
        "    });\n"
        "    var backBtn = card.querySelector('.back-btn');\n"
        "    if (backBtn) {\n"
        "      backBtn.addEventListener('click', function() {\n"
        "        showCard(parseInt(this.dataset.goto));\n"
        "      });\n"
        "    }\n"
        "    var nextBtn = card.querySelector('.next-btn');\n"
        "    if (nextBtn) {\n"
        "      nextBtn.addEventListener('click', function() {\n"
        "        showCard(parseInt(this.dataset.goto));\n"
        "      });\n"
        "    }\n"
        "  });\n"
        "\n"
        f"  var FACTORS = {factors_js};\n"
        "  var form = document.getElementById('ratingForm');\n"
        "  if (form) {\n"
        "    form.addEventListener('submit', function(e) {\n"
        "      var missing = FACTORS.filter(function(f) {\n"
        "        return !document.querySelector('input[name=\"rating_' + f + '\"]:checked');\n"
        "      });\n"
        "      if (missing.length > 0) {\n"
        "        e.preventDefault();\n"
        "        showCard(FACTORS.indexOf(missing[0]));\n"
        "        var msg = document.getElementById('errorMsg');\n"
        "        if (msg) msg.style.display = 'block';\n"
        "      }\n"
        "    });\n"
        "  }\n"
        "})();\n"
        "</script>"
    )

    return deck_html + "\n" + js


# ── Page helpers ──────────────────────────────────────────────────────────────

def _render_page(title: str, body_html: str, progress: tuple[int, int] | None = None) -> str:
    """Wrap body_html in a complete HTML document with shared CSS.

    progress: (current, total) — if provided, a progress bar is shown at the top.
    Returns a complete HTML string served directly by Flask (not XMLQuestion-wrapped).
    """
    progress_html = ""
    if progress is not None:
        current, total = progress
        pct = int(100 * current / total) if total > 0 else 0
        progress_html = (
            f'<p class="progress-label">Conversation {current} of {total}</p>'
            f'<div class="progress-bar-wrap">'
            f'<div class="progress-bar-fill" style="width:{pct}%"></div>'
            f"</div>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape_html(title)}</title>
<style>
{_CSS}
</style>
</head>
<body>
{progress_html}{body_html}
{_JS_VALIDATION}
</body>
</html>"""


def _generate_completion_code(prolific_pid: str, study_id: str) -> str:
    """Return the study completion code.

    If PROLIFIC_COMPLETION_CODE is set, returns that fixed code for all participants
    (required for Prolific URL-redirect completion — set this to match whatever code
    you configure in the Prolific study settings).
    Otherwise falls back to a per-participant HMAC code (manual code-entry mode).
    """
    from pipeline.prolific.config import PROLIFIC_COMPLETION_CODE
    if PROLIFIC_COMPLETION_CODE:
        return PROLIFIC_COMPLETION_CODE
    return hmac.new(
        FLASK_SECRET_KEY.encode(),
        f"{prolific_pid}:{study_id}".encode(),
        hashlib.sha256,
    ).hexdigest()[:8].upper()


def _error_page(msg: str, status: int = 400) -> tuple[str, int]:
    """Return an (html, status_code) tuple for an error page."""
    body = (
        f"<h1>Survey Error</h1>"
        f'<p style="color:#c0392b;">{escape_html(msg)}</p>'
        f"<p>Please return to Prolific and contact the researcher if this problem persists.</p>"
    )
    return _render_page("Error", body), status


def _decode_session_order(raw: str | None) -> list[dict]:
    """Decode the items_assigned JSON blob from the session row."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return []


# ── Routes ────────────────────────────────────────────────────────────────────

def _route_survey_start():
    """GET /survey — entry point from Prolific redirect URL."""
    prolific_pid = request.args.get("PROLIFIC_PID", "").strip()
    study_id = request.args.get("STUDY_ID", "").strip()

    if not prolific_pid or not study_id:
        return _error_page(
            "Missing required parameters PROLIFIC_PID and/or STUDY_ID. "
            "Please use the link provided by Prolific."
        )

    # Check for an existing session
    existing = get_session(prolific_pid, study_id)
    if existing:
        if existing["status"] == "complete":
            session["prolific_pid"] = prolific_pid
            session["study_id"] = study_id
            session["session_id"] = existing["session_id"]
            return redirect("/survey/complete")

        # Resume in-progress session
        session["prolific_pid"] = prolific_pid
        session["study_id"] = study_id
        session["session_id"] = existing["session_id"]

        training_done = existing.get("training_completed", 0)
        items_done = existing.get("items_completed", 0)
        if not training_done:
            return redirect("/survey/training/1")
        next_n = min(items_done + 1, ITEMS_PER_SESSION)
        return redirect(f"/survey/item/{next_n}")

    # New session — assign items and create session record
    sample_items, gold_items = assign_items_for_session(prolific_pid, study_id)
    ordered = build_session_order(sample_items, gold_items, prolific_pid)
    gold_ids = [it["behavioral_response_id"] for it in gold_items]

    new_session_id = str(uuid.uuid4())
    create_session(
        prolific_pid=prolific_pid,
        study_id=study_id,
        session_id=new_session_id,
        items=ordered,      # list of enriched item dicts
        gold_items=gold_ids,
    )

    session["prolific_pid"] = prolific_pid
    session["study_id"] = study_id
    session["session_id"] = new_session_id

    return redirect("/survey/training/1")


def _route_training_get(n: int):
    """GET /survey/training/<n> (n = 1 or 2)."""
    prolific_pid = session.get("prolific_pid")
    study_id = session.get("study_id")
    if not prolific_pid or not study_id:
        return redirect("/survey")

    training_items = get_training_items()
    if not training_items:
        # No gold items available yet — skip straight to main survey
        return redirect("/survey/item/1")
    if n < 1 or n > min(TRAINING_ITEMS, len(training_items)):
        return _error_page(f"Invalid training item number: {n}"), 404

    item = training_items[n - 1]
    # Mixed keying so participants learn both forward and reverse-keyed phrasings
    training_keying = "FRFRF"
    conversation_html = render_conversation(item["prompt_data"], item["response_text"])
    sequential_html = _render_ratings_sequential(
        training_keying, submit_label="Submit &amp; See Feedback"
    )

    body = f"""
<h1>Practice Round ({n} of {TRAINING_ITEMS})</h1>
<div class="instructions">
  Please carefully read the conversation between a user and an AI. Then, rate 5 short statements
  about your perspective on <em>how</em> the AI communicated in the <strong>highlighted response</strong>.
  We want your honest perspective. Thank you for helping us understand how AI communicates.
  <br><br>
  This practice round includes reference ratings to help calibrate your responses.
</div>

<h2>Conversation</h2>
<div class="conversation">
{conversation_html}
</div>

<form id="ratingForm" method="POST">
  {sequential_html}
  <input type="hidden" name="start_time" id="start_time_field" value="">
  <p class="error-msg" id="errorMsg">Please rate all statements before submitting.</p>
</form>
<script>document.getElementById("start_time_field").value = Date.now();</script>
"""
    return _render_page(f"Practice Round ({n} of {TRAINING_ITEMS})", body)


def _route_training_post(n: int):
    """POST /survey/training/<n>."""
    prolific_pid = session.get("prolific_pid")
    study_id = session.get("study_id")
    if not prolific_pid or not study_id:
        return redirect("/survey")

    training_items = get_training_items()
    if not training_items or n < 1 or n > min(TRAINING_ITEMS, len(training_items)):
        return _error_page(f"Invalid training item number: {n}")

    item = training_items[n - 1]
    ground_truth = item.get("ground_truth") or {}
    feedback_dict = item.get("feedback") or {}
    conversation_html = render_conversation(item["prompt_data"], item["response_text"])

    # Parse submitted ratings
    participant_ratings: dict[str, int] = {}
    for factor in FACTOR_ORDER:
        raw = request.form.get(f"rating_{factor}", "")
        try:
            val = int(raw)
            if 1 <= val <= 5:
                participant_ratings[factor] = val
        except (TypeError, ValueError):
            pass

    # Build per-statement feedback cards using mixed keying (FRFRF)
    training_keying = "FRFRF"
    fb_cards: list[str] = []
    for i, factor in enumerate(FACTOR_ORDER):
        key = training_keying[i]
        statement = RATING_STATEMENTS[factor][key]
        p_val = participant_ratings.get(factor)
        e_val = ground_truth.get(factor)  # ground_truth is in corrected (forward) space
        explanation = feedback_dict.get(factor, "")

        # For comparison, convert participant's raw rating to corrected space
        p_corrected = (6 - p_val) if (p_val is not None and key == "R") else p_val
        # For display, show participant's raw rating label as submitted
        p_label = escape_html(_RATING_LABELS[p_val]) if p_val is not None else "&mdash;"
        # For expected label, show what participant SHOULD have picked in raw terms
        # ground_truth is in corrected space; for R-keyed, un-reverse to get expected raw
        if e_val is not None:
            e_raw = (6 - e_val) if key == "R" else e_val
            e_label = escape_html(_RATING_LABELS[e_raw]) if e_raw in _RATING_LABELS else "&mdash;"
        else:
            e_label = "&mdash;"

        if p_corrected is None or e_val is None:
            fb_cls, fb_text = "fb-close", "&mdash;"
        else:
            diff = abs(p_corrected - e_val)
            if diff == 0:
                fb_cls = "fb-perfect"
                fb_text = "Exact match!"
            elif diff == 1:
                fb_cls = "fb-close"
                fb_text = f"Close! {escape_html(explanation)}".strip() if explanation else "Close!"
            else:
                fb_cls = "fb-miss"
                fb_text = f"Not quite. {escape_html(explanation)}".strip() if explanation else "Not quite."

        fb_cards.append(
            f'<tr style="border-bottom:1px solid #eee;">'
            f'<td style="padding:10px 8px;color:#444;">{escape_html(statement)}</td>'
            f'<td style="padding:10px 8px;text-align:center;">{p_label}</td>'
            f'<td style="padding:10px 8px;text-align:center;">{e_label}</td>'
            f'<td class="fb-feedback {fb_cls}" style="padding:10px 8px;">'
            f'<strong>{fb_text}</strong></td>'
            f'</tr>'
        )

    # Determine "Continue" link
    last_training = n >= TRAINING_ITEMS or n >= len(training_items)
    if last_training:
        existing = get_session(prolific_pid, study_id)
        if existing:
            update_session_progress(
                prolific_pid=prolific_pid,
                study_id=study_id,
                items_completed=0,
                training_completed=1,
            )
        continue_url = "/survey/item/1"
        continue_label = "Start Survey &rarr;"
    else:
        continue_url = f"/survey/training/{n + 1}"
        continue_label = "Next Practice Item &rarr;"

    body = f"""
<h1>Practice Round ({n} of {TRAINING_ITEMS}): Feedback</h1>

<h2>Conversation</h2>
<div class="conversation">
{conversation_html}
</div>

<table style="width:100%;border-collapse:collapse;margin-bottom:24px;font-size:14px;">
<thead>
  <tr style="background:#f5f5f5;">
    <th style="padding:8px;text-align:left;color:#555;font-size:12px;">Statement</th>
    <th style="padding:8px;text-align:center;width:13%;color:#555;font-size:12px;">Your Rating</th>
    <th style="padding:8px;text-align:center;width:13%;color:#555;font-size:12px;">Researcher Rating</th>
    <th style="padding:8px;text-align:left;width:40%;color:#555;font-size:12px;">Feedback</th>
  </tr>
</thead>
<tbody>
{"".join(fb_cards)}
</tbody>
</table>

<div style="text-align:center;">
  <a href="{continue_url}" class="continue-btn">{continue_label}</a>
</div>
"""
    return _render_page(f"Practice Round ({n} of {TRAINING_ITEMS}): Feedback", body)


def _route_item_get(n: int):
    """GET /survey/item/<n> (n = 1 … ITEMS_PER_SESSION)."""
    prolific_pid = session.get("prolific_pid")
    study_id = session.get("study_id")
    if not prolific_pid or not study_id:
        return redirect("/survey")

    db_session = get_session(prolific_pid, study_id)
    if not db_session:
        return redirect("/survey")

    if n < 1 or n > ITEMS_PER_SESSION:
        return _error_page(f"Invalid item number: {n}")

    order = _decode_session_order(db_session.get("items_assigned"))
    if n > len(order):
        return _error_page(f"Item {n} is out of range for this session (only {len(order)} items).")

    item_slot = order[n - 1]  # enriched item dict from build_session_order
    behavioral_response_id = item_slot["behavioral_response_id"]
    keying = item_slot.get("keying") or generate_keying(behavioral_response_id)
    prompt_id = item_slot.get("prompt_id", "")
    prompt_data = BEHAVIORAL_PROMPTS.get(prompt_id, {})

    # Load response text from main DB
    from pipeline.storage import DB_PATH
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(DB_PATH)
    conn.row_factory = _sqlite3.Row
    try:
        row = conn.execute(
            "SELECT raw_response FROM behavioral_responses WHERE id = ?",
            (behavioral_response_id,),
        ).fetchone()
        response_text = row["raw_response"] if row else None
    finally:
        conn.close()

    if response_text is None:
        return _error_page(
            f"Could not load response {behavioral_response_id} from database."
        )

    conversation_html = render_conversation(prompt_data, response_text)
    is_last = n >= ITEMS_PER_SESSION
    submit_label = "Submit Ratings" if is_last else "Next Conversation &rarr;"
    sequential_html = _render_ratings_sequential(keying, submit_label=submit_label)

    body = f"""
<div class="instructions">
  Please carefully read the conversation between a user and an AI. Then, rate 5 short statements
  about your perspective on <em>how</em> the AI communicated in the <strong>highlighted response</strong>.
  We want your honest perspective. Thank you for helping us understand how AI communicates.
</div>

<h2>Conversation</h2>
<div class="conversation">
{conversation_html}
</div>

<form id="ratingForm" method="POST">
  {sequential_html}
  <input type="hidden" name="behavioral_response_id" value="{behavioral_response_id}">
  <input type="hidden" name="keying" value="{escape_html(keying)}">
  <input type="hidden" name="start_time" id="start_time_field" value="">
  <p class="error-msg" id="errorMsg">Please rate all statements before submitting.</p>
</form>
<script>document.getElementById("start_time_field").value = Date.now();</script>
"""
    return _render_page("Rate this response", body, progress=(n, ITEMS_PER_SESSION))


def _route_item_post(n: int):
    """POST /survey/item/<n>."""
    prolific_pid = session.get("prolific_pid")
    study_id = session.get("study_id")
    if not prolific_pid or not study_id:
        return redirect("/survey")

    db_session = get_session(prolific_pid, study_id)
    if not db_session:
        return redirect("/survey")

    # Parse form
    try:
        behavioral_response_id = int(request.form.get("behavioral_response_id", ""))
    except (TypeError, ValueError):
        return _error_page("Missing or invalid behavioral_response_id.")

    keying = request.form.get("keying", "F" * len(FACTOR_ORDER))
    start_time_str = request.form.get("start_time", "")

    # Compute response time
    response_time: float = 0.0
    try:
        start_ms = float(start_time_str)
        elapsed = (time.time() * 1000 - start_ms) / 1000.0
        if 0 < elapsed < 3600:
            response_time = elapsed
    except (TypeError, ValueError):
        pass

    # Parse raw ratings
    raw_scores: dict[str, int] = {}
    for factor in FACTOR_ORDER:
        raw_val = request.form.get(f"rating_{factor}", "")
        try:
            val = int(raw_val)
            if 1 <= val <= 5:
                raw_scores[factor] = val
        except (TypeError, ValueError):
            pass

    if len(raw_scores) != len(FACTOR_ORDER):
        return _error_page(
            "Incomplete ratings. Please go back and rate all 5 factors."
        )

    corrected_scores = reverse_score(raw_scores, keying)

    # Determine is_gold (server-side only — not from form)
    gold_ids_raw = db_session.get("gold_items_assigned") or "[]"
    try:
        gold_ids: list[int] = json.loads(gold_ids_raw)
    except (json.JSONDecodeError, TypeError):
        gold_ids = []
    is_gold = behavioral_response_id in gold_ids

    # prompt_id from session order
    order = _decode_session_order(db_session.get("items_assigned"))
    prompt_id = ""
    if 0 < n <= len(order):
        prompt_id = order[n - 1].get("prompt_id", "")

    record_rating(
        prolific_pid=prolific_pid,
        study_id=study_id,
        session_id=db_session["session_id"],
        behavioral_response_id=behavioral_response_id,
        prompt_id=prompt_id,
        keying=keying,
        is_gold=1 if is_gold else 0,
        item_position=n,
        raw_scores=raw_scores,
        corrected_scores=corrected_scores,
        response_time_seconds=response_time,
    )

    update_session_progress(
        prolific_pid=prolific_pid,
        study_id=study_id,
        items_completed=n,
    )

    if n >= ITEMS_PER_SESSION:
        # Session complete — compute gold accuracy and finish
        gold_accuracy = _compute_gold_accuracy(prolific_pid, gold_ids)
        completion_code = _generate_completion_code(prolific_pid, study_id)
        complete_session(
            prolific_pid=prolific_pid,
            study_id=study_id,
            completion_code=completion_code,
            gold_accuracy=gold_accuracy if gold_accuracy is not None else 0.0,
        )
        return redirect("/survey/complete")

    return redirect(f"/survey/item/{n + 1}")


def _compute_gold_accuracy(prolific_pid: str, gold_ids: list[int]) -> float | None:
    """Compute this participant's gold-item accuracy from the DB."""
    if not gold_ids:
        return None

    gold_items = load_gold_items()
    if not gold_items:
        return None

    gold_by_id = {g["behavioral_response_id"]: g for g in gold_items}

    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT behavioral_response_id, corrected_RE, corrected_DE, "
            "corrected_BO, corrected_GU, corrected_VB "
            "FROM prolific_ratings WHERE prolific_pid = ? AND is_gold = 1",
            (prolific_pid,),
        ).fetchall()
    finally:
        conn.close()

    total = correct = 0
    for row in rows:
        rid = row["behavioral_response_id"]
        if rid not in gold_by_id:
            continue
        gt = gold_by_id[rid]["ground_truth"]
        for factor in FACTOR_ORDER:
            val = row[f"corrected_{factor}"]
            if val is None:
                continue
            total += 1
            if abs(val - gt[factor]) <= 1:
                correct += 1

    if total == 0:
        return None
    return correct / total


def _route_complete():
    """GET /survey/complete."""
    prolific_pid = session.get("prolific_pid")
    study_id = session.get("study_id")

    if not prolific_pid or not study_id:
        body = (
            "<h1>Survey Complete</h1>"
            "<p>Thank you for participating!</p>"
            "<p>Please return to Prolific to confirm your submission.</p>"
        )
        return _render_page("Survey Complete", body)

    db_session = get_session(prolific_pid, study_id)
    if db_session and db_session.get("completion_code"):
        completion_code = db_session["completion_code"]
    else:
        completion_code = _generate_completion_code(prolific_pid, study_id)

    prolific_return_url = f"https://app.prolific.com/submissions/complete?cc={completion_code}"

    body = f"""
<h1>Thank You!</h1>
<p>You have completed the survey. Your ratings have been recorded.</p>

<h2>Your Completion Code</h2>
<p style="font-size:14px;color:#555;">
  Copy this code and paste it into Prolific, or click the link below to return automatically.
</p>
<div class="code-box">{escape_html(completion_code)}</div>
<div style="text-align:center;">
  <a href="{prolific_return_url}" class="continue-btn">
    Click here to return to Prolific &rarr;
  </a>
</div>
"""
    return _render_page("Survey Complete", body)


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> Flask:
    """Create and configure the Flask application.

    Usage:
        app = create_app()
        app.run(host="0.0.0.0", port=5000, debug=True)
    """
    app = Flask(__name__)
    app.secret_key = FLASK_SECRET_KEY

    @app.route("/survey")
    def survey_start():
        return _route_survey_start()

    @app.route("/survey/training/<int:n>", methods=["GET"])
    def training_get(n: int):
        return _route_training_get(n)

    @app.route("/survey/training/<int:n>", methods=["POST"])
    def training_post(n: int):
        return _route_training_post(n)

    @app.route("/survey/item/<int:n>", methods=["GET"])
    def item_get(n: int):
        return _route_item_get(n)

    @app.route("/survey/item/<int:n>", methods=["POST"])
    def item_post(n: int):
        return _route_item_post(n)

    @app.route("/survey/complete")
    def survey_complete():
        return _route_complete()

    @app.route("/health")
    def health():
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    return app
