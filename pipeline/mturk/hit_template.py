"""Render self-contained HTML HITs for MTurk behavioral rating tasks."""

from __future__ import annotations

import random

from pipeline.judge_prompt import FACTOR_ORDER, FACTOR_NAMES, RATING_STATEMENTS
from pipeline.mturk.config import MTURK_SEED


def generate_keying(response_id: int) -> str:
    """Return a deterministic 5-char F/R keying string seeded by (MTURK_SEED, response_id).

    All 3 raters for the same HIT see the same statement versions — keying is
    per-HIT, not per-worker.
    """
    rng = random.Random(str((MTURK_SEED, response_id)))
    return "".join(rng.choice("FR") for _ in FACTOR_ORDER)


def render_hit_html(
    response_text: str,
    prompt_data: dict,
    keying: str,
    response_id: int,
    is_gold: bool = False,
) -> str:
    """Render a complete MTurk HIT as an XMLQuestion-wrapped HTML string.

    Args:
        response_text: The AI assistant's raw response text to rate.
        prompt_data:   Behavioral prompt dict (text, or turn1_*/turn2_* fields).
        keying:        5-char F/R string (one char per FACTOR_ORDER).
        response_id:   behavioral_response_id — embedded as hidden field.
        is_gold:       Whether this is a gold standard item.

    Returns:
        Full XMLQuestion XML string ready to pass to MTurk CreateHIT.
    """
    conversation_html = render_conversation(prompt_data, response_text)
    ratings_html = render_ratings_form(keying)
    html_body = _render_full_html(conversation_html, ratings_html, keying, response_id, is_gold)

    return (
        '<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas'
        '/2011-11-11/HTMLQuestion.xsd">\n'
        "  <HTMLContent><![CDATA[\n"
        f"{html_body}\n"
        "  ]]></HTMLContent>\n"
        "  <FrameHeight>0</FrameHeight>\n"
        "</HTMLQuestion>"
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def escape_html(text: str) -> str:
    """Minimal HTML escaping for safe inline display."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def render_conversation(prompt_data: dict, response_text: str) -> str:
    """Build the conversation display HTML block."""
    is_two_turn = bool(prompt_data.get("is_two_turn"))
    parts: list[str] = []

    turn_style_user = (
        'style="background:#f0f4ff;border-radius:6px;padding:12px 16px;margin-bottom:8px;"'
    )
    turn_style_ai = (
        'style="background:#f9f9f9;border-radius:6px;padding:12px 16px;margin-bottom:8px;"'
    )
    turn_style_ai_rate = (
        'style="background:#fff8e1;border:2px solid #f0b429;border-radius:6px;'
        'padding:12px 16px;margin-bottom:8px;"'
    )

    if is_two_turn:
        # Turn 1 user
        parts.append(
            f'<div {turn_style_user}>'
            f'<strong>User:</strong><br>'
            f'<span style="white-space:pre-wrap">{escape_html(prompt_data["turn1_user"])}</span>'
            f"</div>"
        )
        # Turn 1 assistant
        parts.append(
            f'<div {turn_style_ai}>'
            f'<strong>AI:</strong><br>'
            f'<span style="white-space:pre-wrap">{escape_html(prompt_data["turn1_assistant"])}</span>'
            f"</div>"
        )
        # Turn 2 user
        parts.append(
            f'<div {turn_style_user}>'
            f'<strong>User:</strong><br>'
            f'<span style="white-space:pre-wrap">{escape_html(prompt_data["turn2_user"])}</span>'
            f"</div>"
        )
        # Turn 2 assistant — rate THIS response
        parts.append(
            f'<div {turn_style_ai_rate}>'
            f'<strong>AI <em>(RATE ONLY THIS RESPONSE)</em>:</strong><br>'
            f'<span style="white-space:pre-wrap">{escape_html(response_text)}</span>'
            f"</div>"
        )
    else:
        # Single-turn user message
        parts.append(
            f'<div {turn_style_user}>'
            f'<strong>User:</strong><br>'
            f'<span style="white-space:pre-wrap">{escape_html(prompt_data["text"])}</span>'
            f"</div>"
        )
        # Single-turn AI response — only response to rate
        parts.append(
            f'<div {turn_style_ai_rate}>'
            f'<strong>AI:</strong><br>'
            f'<span style="white-space:pre-wrap">{escape_html(response_text)}</span>'
            f"</div>"
        )

    return "\n".join(parts)


def render_ratings_form(keying: str) -> str:
    """Build the 5-factor rating table HTML."""
    rows: list[str] = []
    for i, factor in enumerate(FACTOR_ORDER):
        key = keying[i]
        statement = RATING_STATEMENTS[factor][key]
        factor_name = FACTOR_NAMES[factor]

        radio_cells = ""
        for val in range(1, 6):
            radio_cells += (
                f'<td style="text-align:center;padding:4px 8px;">'
                f'<input type="radio" name="rating_{factor}" value="{val}" '
                f'required id="r_{factor}_{val}">'
                f"</td>"
            )

        rows.append(
            f'<tr style="border-bottom:1px solid #eee;">'
            f'<td style="padding:10px 8px;font-size:14px;max-width:420px;">'
            f'<span style="color:#666;font-size:11px;text-transform:uppercase;'
            f'letter-spacing:.05em">{escape_html(factor_name)}</span><br>'
            f"{escape_html(statement)}"
            f"</td>"
            f"{radio_cells}"
            f"</tr>"
        )

    header_cells = (
        '<th style="text-align:center;padding:6px 8px;font-size:12px;color:#555;">'
        "Strongly<br>Disagree</th>"
        '<th style="text-align:center;padding:6px 8px;font-size:12px;color:#555;">2</th>'
        '<th style="text-align:center;padding:6px 8px;font-size:12px;color:#555;">'
        "Neither<br>Agree nor<br>Disagree</th>"
        '<th style="text-align:center;padding:6px 8px;font-size:12px;color:#555;">4</th>'
        '<th style="text-align:center;padding:6px 8px;font-size:12px;color:#555;">'
        "Strongly<br>Agree</th>"
    )

    return (
        f'<table style="width:100%;border-collapse:collapse;">'
        f"<thead><tr>"
        f'<th style="text-align:left;padding:6px 8px;font-size:12px;color:#555;">Statement</th>'
        f"{header_cells}"
        f"</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
    )


def _render_full_html(
    conversation_html: str,
    ratings_html: str,
    keying: str,
    response_id: int,
    is_gold: bool,
) -> str:
    """Assemble the complete HTML document."""
    js_validation = _render_js_validation()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rate an AI Assistant's Response</title>
<style>
  body {{
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px 24px 40px;
    color: #222;
    background: #fff;
    font-size: 15px;
    line-height: 1.5;
  }}
  h1 {{ font-size: 20px; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; margin-top: 24px; margin-bottom: 8px; color: #333; }}
  .instructions {{
    background: #f5f5f5;
    border-left: 4px solid #4a90e2;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin-bottom: 24px;
    font-size: 14px;
  }}
  .instructions ul {{ margin: 8px 0 0 0; padding-left: 20px; }}
  .instructions li {{ margin-bottom: 4px; }}
  .conversation {{ margin-bottom: 24px; }}
  .scale-note {{
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #888;
    margin-bottom: 4px;
    padding: 0 8px;
  }}
  .submit-btn {{
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
  }}
  .submit-btn:hover {{ background: #357abd; }}
  .error-msg {{
    color: #c0392b;
    font-size: 14px;
    text-align: center;
    margin-top: 10px;
    display: none;
  }}
</style>
</head>
<body>
<h1>Rate an AI Assistant's Response</h1>

<div class="instructions">
  <strong>Instructions:</strong>
  <ul>
    <li>Read the conversation below carefully, focusing on <em>how the AI communicates</em>.</li>
    <li>Rate the highlighted AI response on each of the 5 statements below.</li>
    <li>Use the 1&ndash;5 scale: <strong>1 = Strongly Disagree</strong>, <strong>5 = Strongly Agree</strong>.</li>
    <li>There are no right or wrong answers — rate your genuine impression.</li>
    <li>This task takes about 3&ndash;5 minutes.</li>
  </ul>
</div>

<h2>Conversation</h2>
<div class="conversation">
{conversation_html}
</div>

<h2>Your Ratings</h2>
<p style="font-size:13px;color:#555;margin-bottom:12px;">
  Rate the highlighted AI response on each statement below.
  The yellow box marks the response you are rating.
</p>
<div class="scale-note">
  <span>1 = Strongly Disagree</span>
  <span>5 = Strongly Agree</span>
</div>

<form id="ratingForm" method="POST">
  {ratings_html}

  <!-- Hidden fields -->
  <input type="hidden" name="keying" value="{escape_html(keying)}">
  <input type="hidden" name="behavioral_response_id" value="{response_id}">
  <input type="hidden" name="is_gold" value="{'1' if is_gold else '0'}">

  <button type="submit" class="submit-btn">Submit Ratings</button>
  <p class="error-msg" id="errorMsg">
    Please rate all 5 statements before submitting.
  </p>
</form>

{js_validation}
</body>
</html>"""


def _render_js_validation() -> str:
    """Return inline JS that validates all 5 factors are rated before submit."""
    factors_js = "[" + ", ".join(f'"{f}"' for f in FACTOR_ORDER) + "]"
    return f"""<script>
(function() {{
  var FACTORS = {factors_js};
  document.getElementById("ratingForm").addEventListener("submit", function(e) {{
    var missing = FACTORS.filter(function(f) {{
      return !document.querySelector('input[name="rating_' + f + '"]:checked');
    }});
    if (missing.length > 0) {{
      e.preventDefault();
      document.getElementById("errorMsg").style.display = "block";
    }}
  }});
}})();
</script>"""
