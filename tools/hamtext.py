"""Amateur radio help for the transcripts: a Whisper prompt, a callsign parser and validation."""
import functools
import os
import re

# Whisper takes this as the text before the audio, so it prefers these words and spellings
PROMPT = ("CQ contest, CQ contest, this is Lima Zulu Nine Whiskey. QRZ? Five nine, one four. "
          "Roger, thank you, seven three. Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India "
          "Juliett Kilo Lima Mike November Oscar Papa Quebec Romeo Sierra Tango Uniform Victor "
          "Whiskey X-ray Yankee Zulu.")

NATO = {"alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E", "foxtrot": "F",
        "fox": "F", "golf": "G", "hotel": "H", "india": "I", "juliet": "J", "juliett": "J", "kilo": "K",
        "lima": "L", "lema": "L", "mike": "M", "november": "N", "oscar": "O", "papa": "P", "quebec": "Q",
        "romeo": "R", "sierra": "S", "tango": "T", "uniform": "U", "victor": "V", "whiskey": "W",
        "whisky": "W", "xray": "X", "x-ray": "X", "yankee": "Y", "zulu": "Z"}
DIGITS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
          "seven": "7", "eight": "8", "nine": "9", "niner": "9"}
# ITU style: prefix (1-2 letters, or digit+letter, or letter+digit), one digit, suffix of 1-4 letters
CALL = re.compile(r"^(?:[A-Z]{1,2}|[0-9][A-Z]|[A-Z][0-9])[0-9][A-Z]{1,4}$")


def symbols(text):
    """The text as a list of words, where NATO words and numbers become letters and digits.

    A normal word or a full stop is None, and a comma is ",".
    """
    text = re.sub(r"([a-z])([A-Z0-9])", r"\1 \2", text)          # LimaZulu9 -> Lima Zulu 9
    text = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", text)
    out = []
    for w in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?|[0-9]+|[,.;:!?]", text):
        lw = w.lower()
        if w == ",":
            out.append(",")
        elif not w[0].isalnum():
            out.append(None)
        elif lw in NATO:
            out.append(NATO[lw])
        elif lw in DIGITS:
            out.append(DIGITS[lw])
        elif w.isdigit():
            out.extend(w)
        elif re.fullmatch(r"[A-Z0-9]{3,7}", w) and any(c.isdigit() for c in w):
            out.append(w)                                         # already a callsign, for example TM5C
        else:
            out.append(None)
    return out


def _best(seq):
    """The longest valid callsign in a string of letters and digits."""
    best = ""
    for i in range(len(seq)):
        for j in range(len(seq), i + 2, -1):
            if j - i > len(best) and CALL.match(seq[i:j]):
                best = seq[i:j]
                break
    return best


def callsigns(text):
    """Callsigns spelled in the text, for example 'Lima Zulu 9 Whiskey' -> LZ9W.

    Whisper often puts commas between spelled letters, so a comma ends a callsign
    only if the parts give a valid callsign: 'Tango, Mike 5, Charlie' -> TM5C.
    """
    found = []

    def add(c):
        if c and c not in found:
            found.append(c)

    runs, run = [], []
    for s in symbols(text) + [None]:
        if s is not None and len(s) > 1 and s != ",":
            if CALL.match(s):
                add(s)
            s = None
        if s is None:
            runs.append(run)
            run = []
        else:
            run.append(s)
    for run in runs:
        parts = [p for p in "".join(run).split(",") if p]
        n = len(found)
        i = 0
        while i < len(parts):
            # the longest join of whole parts that is a callsign: 'Y', 'P8T' -> YP8T,
            # but a repeated callsign stays two: 'LZ9W', 'LZ9W' -> LZ9W
            for j in range(len(parts), i, -1):
                if CALL.match("".join(parts[i:j])):
                    add("".join(parts[i:j]))
                    i = j
                    break
            else:
                add(_best(parts[i]))
                i += 1
        if len(found) == n:
            add(_best("".join(parts)))
    return found


# --- validation with the Super Check Partial list (active contest callsigns) and cty.dat (prefixes)

# other phonetic words that operators use, mostly in contests; they are also normal words,
# so a callsign from them is used only if it is in the contest list
ALT = {"america": "A", "amsterdam": "A", "boston": "B", "baltimore": "B", "canada": "C", "denmark": "D",
       "england": "E", "france": "F", "germany": "G", "honolulu": "H", "italy": "I", "japan": "J",
       "kilowatt": "K", "kentucky": "K", "london": "L", "mexico": "M", "norway": "N", "nancy": "N",
       "ontario": "O", "portugal": "P", "peru": "P", "queen": "Q", "radio": "R", "santiago": "S",
       "sugar": "S", "tokyo": "T", "texas": "T", "united": "U", "uruguay": "U", "victoria": "V",
       "washington": "W", "yokohama": "Y", "zanzibar": "Z"}

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_scp, _by_len, _prefix, _exact = set(), {}, {}, {}


def _load():
    if _scp:
        return
    try:
        for line in open(os.path.join(_DATA, "MASTER.SCP"), errors="replace"):
            c = line.strip().upper()
            if c and not c.startswith(("#", "!")):
                _scp.add(c)
                _by_len.setdefault(len(c), []).append(c)
    except FileNotFoundError:
        pass
    try:
        country = None
        for line in open(os.path.join(_DATA, "cty.dat"), errors="replace"):
            if not line.startswith((" ", "\t")):
                country = line.split(":")[0].strip()
                continue
            for p in line.strip().rstrip(";").split(","):
                p = re.sub(r"[(\[<{~].*", "", p.strip())       # drop the zone and other overrides
                if p.startswith("="):
                    _exact[p[1:]] = country
                elif p:
                    _prefix[p] = country
    except FileNotFoundError:
        pass


def country(call):
    """The country of a callsign from the longest matching prefix, or None."""
    _load()
    if call in _exact:
        return _exact[call]
    for n in range(min(len(call), 5), 0, -1):
        if call[:n] in _prefix:
            return _prefix[call[:n]]
    return None


def _subseq(short, long):
    it = iter(long)
    return all(c in it for c in short)


def _one_sub(a, b):
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1


def _digits(c):
    return re.sub(r"\D", "", c)


def check(call, context=()):
    """Validate one callsign. Returns (callsign, status, alternatives).

    status: "known" (in the contest list), "fixed" (corrected to a callsign of the list),
    "new" (valid prefix, not in the list) or None (no valid prefix).
    """
    _load()
    if call in _scp:
        return call, "known", []
    cands = _candidates(call)
    ctx = [c for c in cands if c in context]
    if len(ctx) == 1:
        return ctx[0], "fixed", []
    # without context, only missing characters are corrected: a wrong character that Whisper
    # heard is more often a real callsign that is not in the list
    longer = [c for c in cands if len(c) > len(call)]
    if len(longer) == 1:
        return longer[0], "fixed", []
    if country(call):
        return call, "new", sorted(cands, key=lambda c: abs(len(c) - len(call)))[:4]
    return call, None, []


@functools.lru_cache(maxsize=20000)
def _candidates(call):
    """Callsigns of the contest list with 1-2 missing, 1 wrong or 1 extra character, and the same digits."""
    cands = []
    for n in (len(call) + 1, len(call) + 2, len(call), len(call) - 1):
        for c in _by_len.get(n, []):
            if _digits(c) != _digits(call):
                continue                       # the digits are usually heard correctly
            if (n > len(call) and _subseq(call, c)) or (n == len(call) and _one_sub(call, c)) \
                    or (n < len(call) and _subseq(c, call)):
                cands.append(c)
    return tuple(cands)


def resolve(text, context=()):
    """Callsigns in the text, validated: a list of {"call", "status", "heard", "country", "alts"}."""
    out, seen = [], set()
    for c in callsigns(text):
        best, status, alts = check(c, context)
        if status and best not in seen:
            seen.add(best)
            out.append({"call": best, "status": status, "heard": c, "country": country(best), "alts": alts})
    # the other phonetic words: only callsigns of the contest list
    alt_text = " ".join(ALT.get(w.lower(), w) if w.lower() in ALT else w for w in re.findall(r"\S+", text))
    if alt_text != text:
        for c in callsigns(re.sub(r"\b([A-Z])\b", lambda m: m.group(1).lower() and _nato_word(m.group(1)), alt_text)):
            if c in _scp and c not in seen:
                seen.add(c)
                out.append({"call": c, "status": "known", "heard": c, "country": country(c), "alts": []})
    return out


_NATO_WORD = {v: k for k, v in NATO.items() if k not in ("alfa", "juliet", "whisky", "xray", "lema", "fox")}


def _nato_word(letter):
    return _NATO_WORD.get(letter, letter)
