"""English-first TTS edge-case corpus.

`reference` is populated ONLY for clean-prose (harvard) cases, where an absolute
WER against the exact text is meaningful. All other categories have
reference=None and are evaluated by RELATIVE WER (config A vs B on the same
text), per the chosen methodology — there is no hand-authored verbalization
oracle for numbers/dates/etc.
"""
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    text: str
    reference: Optional[str] = None  # absolute-WER target (clean prose only)
    notes: str = ""


# category id -> (label, has_reference)
CATEGORY_META = {
    "harvard":    ("Harvard (clean prose)", True),
    "cardinal":   ("Cardinal numbers", False),
    "ordinal":    ("Ordinal numbers", False),
    "decimal":    ("Decimals", False),
    "money":      ("Currency / financial", False),
    "measure":    ("Units / measures", False),
    "fraction":   ("Fractions", False),
    "date":       ("Dates", False),
    "time":       ("Times", False),
    "telephone":  ("Phone numbers", False),
    "electronic": ("URLs / emails", False),
    "address":    ("Addresses", False),
    "homograph":  ("Homographs", False),
    "prosody":    ("Prosody / punctuation", False),
    "expression": ("Expression tags", False),
    "robustness": ("Robustness / abuse", False),
}


CASES: List[Case] = [
    # --- Harvard clean prose (absolute WER) ---
    Case("harvard_01", "harvard", "The birch canoe slid on the smooth planks.",
         reference="the birch canoe slid on the smooth planks"),
    Case("harvard_02", "harvard", "Glue the sheet to the dark blue background.",
         reference="glue the sheet to the dark blue background"),
    Case("harvard_03", "harvard", "It's easy to tell the depth of a well.",
         reference="its easy to tell the depth of a well"),
    Case("harvard_04", "harvard", "The juice of lemons makes fine punch.",
         reference="the juice of lemons makes fine punch"),
    Case("harvard_05", "harvard", "A salt pickle tastes fine with ham.",
         reference="a salt pickle tastes fine with ham"),

    # --- Semiotic classes (relative WER) ---
    Case("cardinal_01", "cardinal", "There were 1,204 attendees at the 2026 conference."),
    Case("cardinal_02", "cardinal", "The population grew from 999 to 1000000 in a decade."),
    Case("ordinal_01", "ordinal", "She finished 1st, he came 2nd, and I was 33rd."),
    Case("ordinal_02", "ordinal", "This is the 21st century, not the 19th."),
    Case("decimal_01", "decimal", "Pi is approximately 3.14159 and e is about 2.71828."),
    Case("money_01", "money",
         "The startup secured $5.2M in venture capital, a huge leap from their initial $450K seed round.",
         notes="README Financial Expression example"),
    Case("money_02", "money", "It cost €19.99 in Berlin but only £12.50 in London."),
    Case("measure_01", "measure",
         "Our drone battery lasts 2.3h when flying at 30kph with full camera payload.",
         notes="README Technical Unit example"),
    Case("measure_02", "measure", "The CPU runs at 3.4GHz and draws 28W under load."),
    Case("fraction_01", "fraction", "Add 1/2 cup of sugar and 3/4 teaspoon of salt."),
    Case("date_01", "date", "The declaration was signed on 07/04/1776."),
    Case("date_02", "date", "The meeting is on December 25th, 2025."),
    Case("time_01", "time", "Let's meet at 3:30 PM, not 11:05 AM."),
    Case("telephone_01", "telephone",
         "You can reach the hotel front desk at (212) 555-0142 ext. 402 anytime.",
         notes="README Phone Number example"),
    Case("electronic_01", "electronic", "Email support@example.com or visit https://example.com/help."),
    Case("address_01", "address", "Ship it to 1600 Pennsylvania Ave NW, Washington, DC 20500."),

    # --- Homographs (relative WER; listen for correct pronunciation) ---
    Case("homograph_01", "homograph",
         "I read the newspaper yesterday, and today I will read it again.",
         notes="read /rEd/ vs /reed/"),
    Case("homograph_02", "homograph",
         "The bass player caught a large bass at the lake.",
         notes="bass /beis/ vs /bass/"),
    Case("homograph_03", "homograph",
         "Please wind the clock before the strong wind arrives.",
         notes="wind /waind/ vs /wind/"),

    # --- Prosody / punctuation ---
    Case("prosody_01", "prosody", "Are you absolutely sure about this?"),
    Case("prosody_02", "prosody", "Wait... I think... maybe we should stop."),
    Case("prosody_03", "prosody", "She said — and I quote — \"never again!\""),
    Case("prosody_04", "prosody", "We need eggs, milk, bread, and butter."),

    # --- Expression tags (observational; preprocessor passes < > through) ---
    Case("expression_01", "expression", "I waited for hours. <sigh> It was a long day."),
    Case("expression_02", "expression", "That was so funny. <laugh> I couldn't stop."),
    Case("expression_03", "expression", "Let me think. <breath> Okay, here's the plan."),

    # --- Robustness / abuse ---
    Case("robustness_01", "robustness", "A", notes="single character"),
    Case("robustness_02", "robustness", "WHY ARE YOU SHOUTING AT ME RIGHT NOW",
         notes="all caps"),
    Case("robustness_03", "robustness", "Wait, what?!?! No way!!!!!",
         notes="repeated punctuation"),
    Case("robustness_04", "robustness", "Hello \U0001f600 world \U0001f30d nice to meet you",
         notes="emoji (stripped by preprocessor)"),
    Case("robustness_05", "robustness",
         "The doctor diagnosed pneumonoultramicroscopicsilicovolcanoconiosis after the scan.",
         notes="very long word"),
    Case("robustness_06", "robustness",
         "This is the first sentence. Here is a second one. And a third follows. "
         "A fourth keeps going. The fifth wraps it up nicely. We add a sixth for good measure. "
         "A seventh sentence extends the paragraph further still. The eighth nears the end. "
         "A ninth sentence is here. The tenth and final sentence concludes the long passage.",
         notes="long paragraph -> exercises chunk_text"),
]


def cases_for(categories: Optional[List[str]] = None) -> List[Case]:
    if not categories:
        return list(CASES)
    wanted = set(categories)
    return [c for c in CASES if c.category in wanted]


def category_payload() -> dict:
    """Categories + their cases, for the /api/corpus endpoint."""
    out = []
    for cat_id, (label, has_ref) in CATEGORY_META.items():
        cases = [asdict(c) for c in CASES if c.category == cat_id]
        out.append({
            "id": cat_id,
            "label": label,
            "has_reference": has_ref,
            "n": len(cases),
            "cases": cases,
        })
    return {"categories": out, "total_cases": len(CASES)}
