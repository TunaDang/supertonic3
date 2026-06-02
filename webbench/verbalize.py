"""Full-semiotic text verbalizer: turn written forms into spoken words.

Opt-in. Used (a) on the WER reference so absolute WER is meaningful on number
categories, and (b) as an optional TTS-input pre-pass so the model reads words
instead of symbols. Runs UPSTREAM of helper.py's _preprocess_text.

num2words alone only does plain numbers/ordinals, so a regex pre-pass detects
money / measures / phones / dates / times / percent / decimals first. Patterns
are ordered most-specific-first; anything unmatched is left untouched.
"""
import re

from num2words import num2words

MAG = {"k": "thousand", "m": "million", "b": "billion", "t": "trillion"}
CUR = {"$": "dollars", "€": "euros", "£": "pounds", "¥": "yen"}
CUR_SUB = {"$": "cents", "€": "cents", "£": "pence", "¥": "sen"}
UNITS = {
    "kph": "kilometers per hour", "mph": "miles per hour", "km/h": "kilometers per hour",
    "km": "kilometers", "cm": "centimeters", "mm": "millimeters", "nm": "nanometers",
    "kg": "kilograms", "mg": "milligrams", "lb": "pounds", "lbs": "pounds", "oz": "ounces",
    "ghz": "gigahertz", "mhz": "megahertz", "khz": "kilohertz", "hz": "hertz",
    "kw": "kilowatts", "w": "watts", "kwh": "kilowatt hours", "v": "volts", "a": "amperes",
    "h": "hours", "hr": "hours", "hrs": "hours", "min": "minutes", "sec": "seconds", "ms": "milliseconds",
    "gb": "gigabytes", "mb": "megabytes", "kb": "kilobytes", "tb": "terabytes",
    "ft": "feet", "in": "inches", "mi": "miles", "yd": "yards",
}
MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
_DIGIT = {str(i): num2words(i) for i in range(10)}


def _say_int(s: str) -> str:
    s = s.replace(",", "")
    return num2words(int(s)) if s else "zero"


def _say_number(s: str) -> str:
    """'5.2' -> 'five point two'; '1,204' -> 'one thousand two hundred four'."""
    s = s.replace(",", "")
    if "." in s:
        intp, frac = s.split(".", 1)
        words = num2words(int(intp)) if intp else "zero"
        digits = " ".join(_DIGIT[d] for d in frac if d.isdigit())
        return f"{words} point {digits}"
    return num2words(int(s))


def _say_digits(s: str) -> str:
    return " ".join(_DIGIT[d] for d in s if d.isdigit())


def _say_year(y: int) -> str:
    if 1100 <= y < 2000 or 2010 <= y < 2100:
        hi, lo = divmod(y, 100)
        lo_words = "oh " + num2words(lo) if 0 < lo < 10 else (num2words(lo) if lo else "hundred")
        return f"{num2words(hi)} {lo_words}".strip()
    return num2words(y)


def _money(m):
    g = m.groups()
    cur, num = g[0], g[1]
    mag = (g[2] if len(g) > 2 and g[2] else "").lower()
    if mag:  # $5.2M -> five point two million dollars
        return f"{_say_number(num)} {MAG[mag]} {CUR[cur]}"
    num = num.replace(",", "")
    if "." in num:  # $19.99 -> nineteen dollars and ninety nine cents
        d, c = num.split(".", 1)
        c = (c + "00")[:2]
        out = f"{_say_int(d)} {CUR[cur]}"
        if int(c):
            out += f" and {num2words(int(c))} {CUR_SUB[cur]}"
        return out
    return f"{_say_int(num)} {CUR[cur]}"


def _time(m):
    h, mn, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    if mn == 0:
        spoken = f"{num2words(h)} o'clock"
    elif mn < 10:
        spoken = f"{num2words(h)} oh {num2words(mn)}"
    else:
        spoken = f"{num2words(h)} {num2words(mn)}"
    if ap:
        spoken += " " + ap.upper().replace(".", "")
    return spoken


def _date(m):
    a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000 if y < 50 else 1900
    if 1 <= a <= 12:  # assume US MM/DD/YYYY
        return f"{MONTHS[a]} {num2words(b, to='ordinal')} {_say_year(y)}"
    return f"{_say_digits(m.group(1))} {_say_digits(m.group(2))} {_say_year(y)}"


def _measure(m):
    num, unit = m.group(1), m.group(2).lower()
    return f"{_say_number(num)} {UNITS[unit]}"


_UNIT_ALT = "|".join(sorted((re.escape(u) for u in UNITS), key=len, reverse=True))

# Order matters: most-specific first.
_RULES = [
    # money with magnitude: $5.2M, €1.3B
    (re.compile(r"([$€£¥])\s?(\d[\d,]*(?:\.\d+)?)\s?([KkMmBbTt])\b"), _money),
    # plain money: $19.99, £12.50, $100
    (re.compile(r"([$€£¥])\s?(\d[\d,]*(?:\.\d+)?)"), _money),
    # phone: (212) 555-0142 / 1-800-555-1234
    (re.compile(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b"), lambda m: _say_digits(m.group(0))),
    (re.compile(r"\b1[\s.\-]\d{3}[\s.\-]\d{3}[\s.\-]\d{4}\b"), lambda m: _say_digits(m.group(0))),
    (re.compile(r"\bext\.?\s?(\d+)", re.I), lambda m: "extension " + _say_digits(m.group(1))),
    # date MM/DD/YYYY
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b"), _date),
    # time 3:30 PM / 14:45
    (re.compile(r"\b(\d{1,2}):(\d{2})\s?([AaPp]\.?[Mm]\.?)?\b"), _time),
    # percent
    (re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?%"), lambda m: _say_number(m.group(1)) + " percent"),
    # measure: number + known unit (e.g. 30kph, 2.3h, 3.4GHz)
    (re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s?(" + _UNIT_ALT + r")\b", re.I), _measure),
    # ordinal: 1st, 21st, 33rd
    (re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.I), lambda m: num2words(int(m.group(1)), to="ordinal")),
    # decimals / cardinals: 3.14159, 1,204, 1000000
    (re.compile(r"\b\d[\d,]*(?:\.\d+)?\b"), lambda m: _say_number(m.group(0))),
]


def verbalize(text: str) -> str:
    """Convert written numeric/semiotic forms to spoken words. Best-effort;
    leaves unmatched text and words untouched."""
    if not text:
        return text
    out = text
    for pat, repl in _RULES:
        out = pat.sub(repl, out)
    out = re.sub(r"\s+", " ", out).strip()
    return out
