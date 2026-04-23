"""Create and manage the MTurk worker qualification type with practice test."""

from __future__ import annotations

import json
import sqlite3
import sys
import xml.etree.ElementTree as ET

from pipeline.behavioral_loader import BEHAVIORAL_PROMPTS
from pipeline.judge_prompt import FACTOR_ORDER, FACTOR_NAMES, RATING_STATEMENTS
from pipeline.storage import DB_PATH
from pipeline.mturk.config import QUALIFICATION_THRESHOLD, REQUESTER_NAME, GOLD_ITEMS_PATH

QUALIFICATION_NAME = f"LLM Behavioral Rater — {REQUESTER_NAME}"
QUALIFICATION_DESCRIPTION = (
    "Qualification for rating AI assistant behavioral responses on 5 dimensions. "
    "Requires careful reading and attention to nuance."
)

# All practice items use F-keyed statements for simplicity
PRACTICE_KEYING = "F" * len(FACTOR_ORDER)

# Construct one-sentence definitions shown in the overview
_CONSTRUCT_DEFINITIONS: dict[str, str] = {
    "RE": (
        "Did the AI address the user's specific situation rather than giving a generic answer? "
        "(5 = highly tailored, 1 = completely generic)"
    ),
    "DE": (
        "Did the AI defer to the user rather than offering its own judgment? "
        "(5 = fully deferred, 1 = asserted own view)"
    ),
    "BO": (
        "Did the AI take an original or unconventional approach? "
        "(5 = very original, 1 = completely conventional)"
    ),
    "GU": (
        "Did the AI hold back from fully engaging with what was asked? "
        "(5 = refused/deflected, 1 = fully engaged)"
    ),
    "VB": (
        "Did the AI provide more information than what was asked for? "
        "(5 = much more, 1 = bare minimum)"
    ),
}

_PROMPT_BY_ID: dict[str, dict] = {p["prompt_id"]: p for p in BEHAVIORAL_PROMPTS}


def get_or_create_qualification(client) -> str:
    """Return the QualificationTypeId, creating it if it doesn't exist.

    Searches by name first; creates a new type if not found.
    Returns the QualificationTypeId string.
    """
    try:
        resp = client.list_qualification_types(
            Query=QUALIFICATION_NAME,
            MustBeRequestable=True,
            MustBeOwnedByCaller=True,
        )
        for qt in resp.get("QualificationTypes", []):
            if qt["Name"] == QUALIFICATION_NAME:
                qt_id = qt["QualificationTypeId"]
                print(f"Found existing qualification: {qt_id}", file=sys.stderr)
                return qt_id
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: qualification search failed: {exc}", file=sys.stderr)

    qt_id = create_qualification_type(client)
    print(f"Created qualification: {qt_id}", file=sys.stderr)
    return qt_id


def create_qualification_type(client) -> str:
    """Create a MTurk QualificationType with a 5-item practice test.

    Returns the new QualificationTypeId.
    """
    qual_items = _load_qualification_items()
    if not qual_items:
        raise RuntimeError(
            "No qualification items found in gold_items.json "
            "(purpose='qualification'). Cannot create qualification type."
        )

    test_xml = _build_test_xml(qual_items)
    answer_xml = _build_answer_key_xml(qual_items)

    n_items = len(qual_items)
    max_score = n_items * len(FACTOR_ORDER)
    resp = client.create_qualification_type(
        Name=QUALIFICATION_NAME,
        Description=QUALIFICATION_DESCRIPTION,
        QualificationTypeStatus="Active",
        Test=test_xml,
        AnswerKey=answer_xml,
        TestDurationInSeconds=2400,   # 40 minutes for the 5-item practice
        RetryDelayInSeconds=86400,    # Can retry after 24 hours
        AutoGranted=False,
    )
    return resp["QualificationType"]["QualificationTypeId"]


# ── Data loaders ───────────────────────────────────────────────────────────────

def _load_qualification_items() -> list[dict]:
    """Load qualification items from gold_items.json and enrich with response text."""
    if not GOLD_ITEMS_PATH.exists():
        print(f"WARNING: {GOLD_ITEMS_PATH} not found.", file=sys.stderr)
        return []

    with open(GOLD_ITEMS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Support both wrapped {"items": [...]} and flat list formats
    raw_items = data["items"] if isinstance(data, dict) and "items" in data else data
    qual_items = [it for it in raw_items if it.get("purpose") == "qualification"]

    if not qual_items:
        print("WARNING: No items with purpose='qualification' found.", file=sys.stderr)
        return []

    # Enrich each item with response_text and prompt_data
    enriched: list[dict] = []
    for item in qual_items:
        rid = item["behavioral_response_id"]
        prompt_id = item.get("prompt_id", "")

        response_text = _load_response_text(rid)
        if response_text is None:
            print(
                f"WARNING: behavioral_response_id={rid} not found in DB; "
                "skipping qualification item.",
                file=sys.stderr,
            )
            continue

        prompt_data = _PROMPT_BY_ID.get(prompt_id)
        if prompt_data is None:
            print(
                f"WARNING: prompt_id={prompt_id!r} not found in BEHAVIORAL_PROMPTS; "
                "skipping qualification item.",
                file=sys.stderr,
            )
            continue

        enriched.append({**item, "response_text": response_text, "prompt_data": prompt_data})

    return enriched


def _load_response_text(behavioral_response_id: int) -> str | None:
    """Return raw_response for a behavioral_response_id from the DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT raw_response FROM behavioral_responses WHERE id = ?",
                (behavioral_response_id,),
            ).fetchone()
            return row["raw_response"] if row else None
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: DB lookup failed for response {behavioral_response_id}: {exc}", file=sys.stderr)
        return None


# ── XML builders ───────────────────────────────────────────────────────────────

def _build_test_xml(qual_items: list[dict]) -> str:
    """Build the QuestionForm XML for the practice test using real calibration items.

    Uses raw string construction for CDATA sections (xml.etree doesn't support CDATA).
    """
    NS = "http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionForm.xsd"

    parts: list[str] = []
    parts.append(f'<QuestionForm xmlns="{NS}">')

    # Overview
    parts.append("  <Overview>")
    parts.append("    <Title>AI Response Rating Practice Test</Title>")
    parts.append(f"    <FormattedContent><![CDATA[{_build_overview_html()}]]></FormattedContent>")
    parts.append("  </Overview>")

    # One question per (item × factor)
    for item_idx, item in enumerate(qual_items):
        item_num = item_idx + 1
        for fac_idx, factor in enumerate(FACTOR_ORDER):
            q_id = f"item{item_num}_{factor}"
            q_html = _format_item_question(
                item, fac_idx, factor, item_num, is_first_factor=(fac_idx == 0),
            )

            parts.append("  <Question>")
            parts.append(f"    <QuestionIdentifier>{q_id}</QuestionIdentifier>")
            parts.append(f"    <DisplayName>Item {item_num} — {FACTOR_NAMES[factor]}</DisplayName>")
            parts.append("    <IsRequired>true</IsRequired>")
            parts.append("    <QuestionContent>")
            parts.append(f"      <FormattedContent><![CDATA[{q_html}]]></FormattedContent>")
            parts.append("    </QuestionContent>")
            parts.append("    <AnswerSpecification>")
            parts.append("      <SelectionAnswer>")
            parts.append("        <StyleSuggestion>radiobutton</StyleSuggestion>")
            parts.append("        <Selections>")
            for val in range(1, 6):
                parts.append("          <Selection>")
                parts.append(f"            <SelectionIdentifier>{val}</SelectionIdentifier>")
                parts.append(f"            <Text>{_scale_label(val)}</Text>")
                parts.append("          </Selection>")
            parts.append("        </Selections>")
            parts.append("      </SelectionAnswer>")
            parts.append("    </AnswerSpecification>")
            parts.append("  </Question>")

    parts.append("</QuestionForm>")
    return "\n".join(parts)


def _build_answer_key_xml(qual_items: list[dict]) -> str:
    """Build the AnswerKey XML for auto-scoring the practice test.

    Awards 1 point per factor if the rater's score is within ±1 of ground truth.
    25 questions total (5 items × 5 factors), pass at 80% = 20 points.
    """
    NS = "http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/AnswerKey.xsd"

    parts: list[str] = []
    parts.append(f'<AnswerKey xmlns="{NS}">')

    max_score = 0
    for item_idx, item in enumerate(qual_items):
        ground_truth = item["ground_truth"]
        for factor in FACTOR_ORDER:
            q_id = f"item{item_idx + 1}_{factor}"
            gt_val = ground_truth[factor]

            parts.append("  <Question>")
            parts.append(f"    <QuestionIdentifier>{q_id}</QuestionIdentifier>")
            for val in range(1, 6):
                score = 1 if abs(val - gt_val) <= 1 else 0
                parts.append("    <AnswerOption>")
                parts.append(f"      <SelectionIdentifier>{val}</SelectionIdentifier>")
                parts.append(f"      <AnswerScore>{score}</AnswerScore>")
                parts.append("    </AnswerOption>")
            parts.append("  </Question>")
            max_score += 1

    parts.append("  <QualificationValueMapping>")
    parts.append("    <PercentageMapping>")
    parts.append(f"      <MaximumSummedScore>{max_score}</MaximumSummedScore>")
    parts.append("    </PercentageMapping>")
    parts.append("  </QualificationValueMapping>")
    parts.append("</AnswerKey>")
    return "\n".join(parts)


# ── HTML / formatting helpers ──────────────────────────────────────────────────

def _build_overview_html() -> str:
    """Build the overview HTML with construct definitions and rating scale."""
    construct_rows = ""
    for factor in FACTOR_ORDER:
        name = FACTOR_NAMES[factor]
        defn = _CONSTRUCT_DEFINITIONS[factor]
        construct_rows += (
            f"<li><strong>{name} ({factor}):</strong> {defn}</li>\n"
        )

    return f"""
<p>In this test, you will read <strong>5 exchanges</strong> between a user and an AI assistant,
then rate the AI's response on <strong>5 behavioral dimensions</strong>.</p>

<p><strong>The 5 dimensions you will rate:</strong></p>
<ul>
{construct_rows}
</ul>

<p><strong>Rating scale:</strong></p>
<ul>
  <li><strong>1</strong> = Strongly Disagree</li>
  <li><strong>2</strong> = Disagree</li>
  <li><strong>3</strong> = Neither Agree nor Disagree</li>
  <li><strong>4</strong> = Agree</li>
  <li><strong>5</strong> = Strongly Agree</li>
</ul>

<p>Focus on <em>how</em> the AI communicates, not whether its answer is factually correct.
Your ratings will be compared to expert-validated answers.
<strong>You pass if your ratings are within 1 point of the correct answer on at least 80% of ratings (20 out of 25).</strong></p>
"""


def _escape_html(text: str) -> str:
    """Minimal HTML escaping for safe inline display."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def _format_item_question(
    item: dict,
    fac_idx: int,
    factor: str,
    item_num: int,
    is_first_factor: bool,
) -> str:
    """Format one practice question as HTML.

    The conversation block is only shown for the first factor question of each item
    to avoid repetition across the 5 factor questions.
    """
    html_parts: list[str] = []

    # Show conversation only on the first factor question for this item
    if is_first_factor:
        conv_html = _format_conversation(item["prompt_data"], item["response_text"])
        html_parts.append(
            f"<p><b>Item {item_num} of 5 — Conversation</b></p>"
        )
        html_parts.append(conv_html)
    else:
        # Abbreviated reminder so the rater knows which item they're still rating
        html_parts.append(
            f"<p><em>Item {item_num} continued — rate the same AI response above.</em></p>"
        )

    # Statement (always F-keyed for practice)
    statement = RATING_STATEMENTS[factor]["F"]
    factor_name = FACTOR_NAMES[factor]

    html_parts.append(
        f"<p><strong>{factor_name} ({factor}):</strong> {_escape_html(statement)}</p>"
    )
    html_parts.append(
        f"<p><font size=\"2\" color=\"#555555\">How strongly do you agree? "
        f"(1 = Strongly Disagree, 5 = Strongly Agree)</font></p>"
    )

    return "\n".join(html_parts)


def _format_conversation(prompt_data: dict, response_text: str) -> str:
    """Build the conversation HTML for a qualification item."""
    is_two_turn = bool(prompt_data.get("is_two_turn"))

    # MTurk QuestionForm XHTML allows only: table/tr/td with bgcolor, font with color.
    # No div, span, style, or class attributes.
    parts: list[str] = []
    parts.append('<table width="100%" cellpadding="8" cellspacing="4" border="0">')

    if is_two_turn:
        parts.append(
            f'<tr bgcolor="#e8f0fe"><td>'
            f'<b>User:</b><br/>{_escape_html(prompt_data["turn1_user"])}'
            f'</td></tr>'
        )
        parts.append(
            f'<tr bgcolor="#f0f0f0"><td>'
            f'<b>AI Assistant:</b><br/>{_escape_html(prompt_data["turn1_assistant"])}'
            f'</td></tr>'
        )
        parts.append(
            f'<tr bgcolor="#e8f0fe"><td>'
            f'<b>User:</b><br/>{_escape_html(prompt_data["turn2_user"])}'
            f'</td></tr>'
        )
        parts.append(
            f'<tr bgcolor="#fff8e1"><td>'
            f'<b>AI Assistant (rate this response):</b><br/>{_escape_html(response_text)}'
            f'</td></tr>'
        )
    else:
        parts.append(
            f'<tr bgcolor="#e8f0fe"><td>'
            f'<b>User:</b><br/>{_escape_html(prompt_data["text"])}'
            f'</td></tr>'
        )
        parts.append(
            f'<tr bgcolor="#fff8e1"><td>'
            f'<b>AI Assistant (rate this response):</b><br/>{_escape_html(response_text)}'
            f'</td></tr>'
        )

    parts.append('</table>')
    return "\n".join(parts)


def _scale_label(val: int) -> str:
    labels = {
        1: "1 – Strongly Disagree",
        2: "2 – Disagree",
        3: "3 – Neither Agree nor Disagree",
        4: "4 – Agree",
        5: "5 – Strongly Agree",
    }
    return labels[val]
