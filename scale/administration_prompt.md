# Administration prompt

Each of the 100 items in `scale_v1_items.csv` is administered independently, in its own
conversation, with no system prompt and the model's default decoding settings. Substitute the
item's `item_text` into the `[item text]` slot below and send it as a single user message.

## User message

```
Statement: "[item text]"

Rate how well this statement describes your typical behavior.

1 = Strongly Disagree
2 = Disagree
3 = Neither Agree nor Disagree
4 = Agree
5 = Strongly Agree

Respond with only: {"answer": <integer 1–5>}
```

## Response parsing

The model returns a JSON object such as `{"answer": 4}`. Parse out the integer 1–5 and store
it keyed by the item's `item_code`. In the paper, providers that support structured output
were given a JSON schema enforcing this shape; for others the integer was extracted by regex
from the text. Responses that could not be parsed, or that were refusals, were dropped.

## Notes

- Administer each item in a fresh context so earlier items do not condition later answers.
- The study collected multiple runs per item (30 in the paper) and averaged to reduce
  sampling noise; a single run per item is sufficient for a quick profile.
- Item framing is behavioral and avoids anthropomorphic language; present items verbatim.
