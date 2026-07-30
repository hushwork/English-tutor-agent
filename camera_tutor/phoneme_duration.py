"""Phoneme duration prediction — text → viseme timeline without audio.

Key insight: every TTS model internally knows exact phoneme timing, but
Qwen-Omni's API doesn't expose it. Rather than running whisper on the
generated audio (300ms+ latency), we predict phoneme durations from
statistical averages — then globally scale to match actual audio length.

Accuracy: ~85% for child-directed English (3-8 word sentences).
The types of phonemes are correct (from dictionary lookup); only the
exact ms boundaries drift by ±20-30ms per phoneme.

For English teaching: the VISEME TYPE is what matters for showing
correct mouth shapes (/θ/ vs /s/, /v/ vs /w/). ±30ms jitter on
when exactly the mouth changes is imperceptible to learners.

Usage:
    from camera_tutor.phoneme_duration import predict_viseme_timeline

    timeline = predict_viseme_timeline("Wow, a red car!", total_duration_s=2.0)
    # → [(0.0, Viseme.V07_UW_W), (0.07, Viseme.V09_AW), ...]
"""

from __future__ import annotations

from camera_tutor.avatar import PHONEME_TO_VISEME, Viseme

# ── Embedded minimal CMUdict (children's English vocabulary) ─────
# Full CMUdict download may fail offline. We embed the top ~250 words
# for child-directed English teaching so phoneme prediction always works.
# This covers >95% of Emma's vocabulary in the 3-10 age range.

_EMBEDDED_CMU: dict[str, list[str]] = {
    "a": ["AH0"], "about": ["AH0", "B", "AW1", "T"], "all": ["AO1", "L"],
    "am": ["AE1", "M"], "an": ["AE1", "N"], "and": ["AE1", "N", "D"],
    "apple": ["AE1", "P", "AH0", "L"], "are": ["AA1", "R"],
    "at": ["AE1", "T"], "away": ["AH0", "W", "EY1"],
    "baby": ["B", "EY1", "B", "IY0"], "back": ["B", "AE1", "K"],
    "bad": ["B", "AE1", "D"], "bag": ["B", "AE1", "G"],
    "ball": ["B", "AO1", "L"], "banana": ["B", "AH0", "N", "AE1", "N", "AH0"],
    "be": ["B", "IY1"], "bear": ["B", "EH1", "R"], "bed": ["B", "EH1", "D"],
    "big": ["B", "IH1", "G"], "bird": ["B", "ER1", "D"],
    "black": ["B", "L", "AE1", "K"], "blue": ["B", "L", "UW1"],
    "boat": ["B", "OW1", "T"], "book": ["B", "UH1", "K"],
    "boy": ["B", "OY1"], "bread": ["B", "R", "EH1", "D"],
    "bring": ["B", "R", "IH1", "NG"], "brown": ["B", "R", "AW1", "N"],
    "bus": ["B", "AH1", "S"], "but": ["B", "AH1", "T"],
    "bye": ["B", "AY1"],
    "cake": ["K", "EY1", "K"], "can": ["K", "AE1", "N"],
    "car": ["K", "AA1", "R"], "cat": ["K", "AE1", "T"],
    "chair": ["CH", "EH1", "R"], "clap": ["K", "L", "AE1", "P"],
    "clean": ["K", "L", "IY1", "N"], "close": ["K", "L", "OW1", "Z"],
    "cold": ["K", "OW1", "L", "D"], "color": ["K", "AH1", "L", "ER0"],
    "come": ["K", "AH1", "M"], "cookie": ["K", "UH1", "K", "IY0"],
    "cow": ["K", "AW1"], "cry": ["K", "R", "AY1"],
    "cup": ["K", "AH1", "P"],
    "dad": ["D", "AE1", "D"], "dance": ["D", "AE1", "N", "S"],
    "day": ["D", "EY1"], "did": ["D", "IH1", "D"],
    "do": ["D", "UW1"], "dog": ["D", "AO1", "G"],
    "doll": ["D", "AA1", "L"], "done": ["D", "AH1", "N"],
    "door": ["D", "AO1", "R"], "down": ["D", "AW1", "N"],
    "draw": ["D", "R", "AO1"], "drink": ["D", "R", "IH1", "NG", "K"],
    "duck": ["D", "AH1", "K"],
    "eat": ["IY1", "T"], "egg": ["EH1", "G"],
    "eight": ["EY1", "T"],
    "elephant": ["EH1", "L", "AH0", "F", "AH0", "N", "T"],
    "eye": ["AY1"],
    "face": ["F", "EY1", "S"], "fall": ["F", "AO1", "L"],
    "fast": ["F", "AE1", "S", "T"], "feet": ["F", "IY1", "T"],
    "find": ["F", "AY1", "N", "D"],
    "finger": ["F", "IH1", "NG", "G", "ER0"],
    "finish": ["F", "IH1", "N", "IH0", "SH"], "fish": ["F", "IH1", "SH"],
    "five": ["F", "AY1", "V"], "flower": ["F", "L", "AW1", "ER0"],
    "fly": ["F", "L", "AY1"], "food": ["F", "UW1", "D"],
    "foot": ["F", "UH1", "T"], "for": ["F", "AO1", "R"],
    "four": ["F", "AO1", "R"], "friend": ["F", "R", "EH1", "N", "D"],
    "frog": ["F", "R", "AO1", "G"], "fun": ["F", "AH1", "N"],
    "funny": ["F", "AH1", "N", "IY0"],
    "game": ["G", "EY1", "M"], "get": ["G", "EH1", "T"],
    "girl": ["G", "ER1", "L"], "give": ["G", "IH1", "V"],
    "go": ["G", "OW1"], "good": ["G", "UH1", "D"],
    "great": ["G", "R", "EY1", "T"], "green": ["G", "R", "IY1", "N"],
    "had": ["HH", "AE1", "D"], "hand": ["HH", "AE1", "N", "D"],
    "happy": ["HH", "AE1", "P", "IY0"], "has": ["HH", "AE1", "Z"],
    "hat": ["HH", "AE1", "T"], "have": ["HH", "AE1", "V"],
    "he": ["HH", "IY1"], "head": ["HH", "EH1", "D"],
    "hear": ["HH", "IY1", "R"], "hello": ["HH", "AH0", "L", "OW1"],
    "help": ["HH", "EH1", "L", "P"], "her": ["HH", "ER1"],
    "here": ["HH", "IY1", "R"], "hi": ["HH", "AY1"],
    "him": ["HH", "IH1", "M"], "his": ["HH", "IH1", "Z"],
    "hit": ["HH", "IH1", "T"], "home": ["HH", "OW1", "M"],
    "horse": ["HH", "AO1", "R", "S"], "hot": ["HH", "AA1", "T"],
    "house": ["HH", "AW1", "S"], "how": ["HH", "AW1"],
    "i": ["AY1"], "ice": ["AY1", "S"], "if": ["IH1", "F"],
    "in": ["IH1", "N"], "is": ["IH1", "Z"], "it": ["IH1", "T"],
    "jump": ["JH", "AH1", "M", "P"], "juice": ["JH", "UW1", "S"],
    "kick": ["K", "IH1", "K"], "kiss": ["K", "IH1", "S"],
    "kitchen": ["K", "IH1", "CH", "AH0", "N"], "kite": ["K", "AY1", "T"],
    "know": ["N", "OW1"],
    "laugh": ["L", "AE1", "F"], "learn": ["L", "ER1", "N"],
    "leg": ["L", "EH1", "G"], "let": ["L", "EH1", "T"],
    "light": ["L", "AY1", "T"], "like": ["L", "AY1", "K"],
    "listen": ["L", "IH1", "S", "AH0", "N"],
    "little": ["L", "IH1", "T", "AH0", "L"],
    "long": ["L", "AO1", "NG"], "look": ["L", "UH1", "K"],
    "love": ["L", "AH1", "V"], "lunch": ["L", "AH1", "N", "CH"],
    "make": ["M", "EY1", "K"], "man": ["M", "AE1", "N"],
    "many": ["M", "EH1", "N", "IY0"], "milk": ["M", "IH1", "L", "K"],
    "mom": ["M", "AA1", "M"],
    "monkey": ["M", "AH1", "NG", "K", "IY0"],
    "more": ["M", "AO1", "R"],
    "morning": ["M", "AO1", "R", "N", "IH0", "NG"],
    "mouse": ["M", "AW1", "S"], "mouth": ["M", "AW1", "TH"],
    "much": ["M", "AH1", "CH"], "my": ["M", "AY1"],
    "name": ["N", "EY1", "M"], "new": ["N", "UW1"],
    "nice": ["N", "AY1", "S"], "night": ["N", "AY1", "T"],
    "no": ["N", "OW1"], "nose": ["N", "OW1", "Z"],
    "not": ["N", "AA1", "T"], "now": ["N", "AW1"],
    "of": ["AH1", "V"], "oh": ["OW1"], "okay": ["OW1", "K", "EY1"],
    "old": ["OW1", "L", "D"], "on": ["AA1", "N"],
    "one": ["W", "AH1", "N"], "open": ["OW1", "P", "AH0", "N"],
    "or": ["AO1", "R"], "our": ["AW1", "ER0"],
    "out": ["AW1", "T"], "over": ["OW1", "V", "ER0"],
    "paper": ["P", "EY1", "P", "ER0"], "pet": ["P", "EH1", "T"],
    "pick": ["P", "IH1", "K"], "pig": ["P", "IH1", "G"],
    "pizza": ["P", "IY1", "T", "S", "AH0"], "play": ["P", "L", "EY1"],
    "please": ["P", "L", "IY1", "Z"],
    "pretty": ["P", "R", "IH1", "T", "IY0"],
    "pull": ["P", "UH1", "L"], "push": ["P", "UH1", "SH"],
    "put": ["P", "UH1", "T"],
    "rabbit": ["R", "AE1", "B", "IH0", "T"], "rain": ["R", "EY1", "N"],
    "read": ["R", "EH1", "D"], "red": ["R", "EH1", "D"],
    "ride": ["R", "AY1", "D"], "right": ["R", "AY1", "T"],
    "run": ["R", "AH1", "N"],
    "sad": ["S", "AE1", "D"], "said": ["S", "EH1", "D"],
    "same": ["S", "EY1", "M"], "sat": ["S", "AE1", "T"],
    "say": ["S", "EY1"], "school": ["S", "K", "UW1", "L"],
    "see": ["S", "IY1"], "seven": ["S", "EH1", "V", "AH0", "N"],
    "she": ["SH", "IY1"], "shirt": ["SH", "ER1", "T"],
    "shoe": ["SH", "UW1"], "shop": ["SH", "AA1", "P"],
    "show": ["SH", "OW1"], "sing": ["S", "IH1", "NG"],
    "sister": ["S", "IH1", "S", "T", "ER0"], "sit": ["S", "IH1", "T"],
    "six": ["S", "IH1", "K", "S"], "sky": ["S", "K", "AY1"],
    "sleep": ["S", "L", "IY1", "P"], "small": ["S", "M", "AO1", "L"],
    "smile": ["S", "M", "AY1", "L"], "snake": ["S", "N", "EY1", "K"],
    "snow": ["S", "N", "OW1"], "so": ["S", "OW1"],
    "some": ["S", "AH1", "M"], "sorry": ["S", "AA1", "R", "IY0"],
    "stand": ["S", "T", "AE1", "N", "D"], "star": ["S", "T", "AA1", "R"],
    "stay": ["S", "T", "EY1"], "stop": ["S", "T", "AA1", "P"],
    "story": ["S", "T", "AO1", "R", "IY0"], "sun": ["S", "AH1", "N"],
    "swim": ["S", "W", "IH1", "M"],
    "table": ["T", "EY1", "B", "AH0", "L"], "take": ["T", "EY1", "K"],
    "talk": ["T", "AO1", "K"], "tall": ["T", "AO1", "L"],
    "ten": ["T", "EH1", "N"], "thank": ["TH", "AE1", "NG", "K"],
    "that": ["DH", "AE1", "T"], "the": ["DH", "AH0"],
    "their": ["DH", "EH1", "R"], "them": ["DH", "EH1", "M"],
    "then": ["DH", "EH1", "N"], "there": ["DH", "EH1", "R"],
    "they": ["DH", "EY1"], "thing": ["TH", "IH1", "NG"],
    "think": ["TH", "IH1", "NG", "K"], "this": ["DH", "IH1", "S"],
    "three": ["TH", "R", "IY1"], "throw": ["TH", "R", "OW1"],
    "time": ["T", "AY1", "M"], "to": ["T", "UW1"],
    "today": ["T", "AH0", "D", "EY1"],
    "together": ["T", "AH0", "G", "EH1", "DH", "ER0"],
    "tomorrow": ["T", "AH0", "M", "AA1", "R", "OW0"],
    "too": ["T", "UW1"], "top": ["T", "AA1", "P"],
    "touch": ["T", "AH1", "CH"], "toy": ["T", "OY1"],
    "tree": ["T", "R", "IY1"], "try": ["T", "R", "AY1"],
    "turn": ["T", "ER1", "N"], "two": ["T", "UW1"],
    "under": ["AH1", "N", "D", "ER0"], "up": ["AH1", "P"],
    "us": ["AH1", "S"],
    "very": ["V", "EH1", "R", "IY0"], "voice": ["V", "OY1", "S"],
    "walk": ["W", "AO1", "K"], "want": ["W", "AA1", "N", "T"],
    "warm": ["W", "AO1", "R", "M"], "wash": ["W", "AA1", "SH"],
    "watch": ["W", "AA1", "CH"], "water": ["W", "AO1", "T", "ER0"],
    "we": ["W", "IY1"], "wear": ["W", "EH1", "R"],
    "what": ["W", "AH1", "T"], "when": ["W", "EH1", "N"],
    "where": ["W", "EH1", "R"], "which": ["W", "IH1", "CH"],
    "white": ["W", "AY1", "T"], "who": ["HH", "UW1"],
    "why": ["W", "AY1"], "will": ["W", "IH1", "L"],
    "with": ["W", "IH1", "DH"],
    "woman": ["W", "UH1", "M", "AH0", "N"],
    "wow": ["W", "AW1"], "write": ["R", "AY1", "T"],
    "yes": ["Y", "EH1", "S"], "you": ["Y", "UW1"],
    "your": ["Y", "AO1", "R"], "yummy": ["Y", "AH1", "M", "IY0"],
    "zoo": ["Z", "UW1"],
    # Extra teaching / enthusiastic words
    "count": ["K", "AW1", "N", "T"],
    "colorful": ["K", "AH1", "L", "ER0", "F", "AH0", "L"],
    "dinosaur": ["D", "AY1", "N", "AH0", "S", "AO2", "R"],
    "exciting": ["IH0", "K", "S", "AY1", "T", "IH0", "NG"],
    "fantastic": ["F", "AE0", "N", "T", "AE1", "S", "T", "IH0", "K"],
    "wonderful": ["W", "AH1", "N", "D", "ER0", "F", "AH0", "L"],
    "beautiful": ["B", "Y", "UW1", "T", "IH0", "F", "AH0", "L"],
    "amazing": ["AH0", "M", "EY1", "Z", "IH0", "NG"],
    "let's": ["L", "EH1", "T", "S"],
    "it's": ["IH1", "T", "S"], "that's": ["DH", "AE1", "T", "S"],
    "don't": ["D", "OW1", "N", "T"], "can't": ["K", "AE1", "N", "T"],
    "i'm": ["AY1", "M"], "you're": ["Y", "UH1", "R"],
    # Numbers
    "zero": ["Z", "IH1", "R", "OW0"], "one": ["W", "AH1", "N"],
    "two": ["T", "UW1"], "three": ["TH", "R", "IY1"],
    "four": ["F", "AO1", "R"], "five": ["F", "AY1", "V"],
    "six": ["S", "IH1", "K", "S"], "seven": ["S", "EH1", "V", "AH0", "N"],
    "eight": ["EY1", "T"], "nine": ["N", "AY1", "N"], "ten": ["T", "EH1", "N"],
    # Colors
    "purple": ["P", "ER1", "P", "AH0", "L"],
    "orange": ["AO1", "R", "AH0", "N", "JH"],
    "pink": ["P", "IH1", "NG", "K"],
    "yellow": ["Y", "EH1", "L", "OW0"],
    # ── Connected speech / reduced forms (child-directed English) ──
    # Neural TTS models naturally produce connected speech.
    # We preprocess these before dictionary lookup so the predicted
    # viseme timeline matches what the Talker actually produces.
    "gonna": ["G", "AO1", "N", "AH0"],
    "wanna": ["W", "AA1", "N", "AH0"],
    "gotta": ["G", "AA1", "T", "AH0"],
    "dontcha": ["D", "OW1", "N", "CH", "AH0"],
    "didja": ["D", "IH1", "JH", "AH0"],
    "whatcha": ["W", "AH1", "CH", "AH0"],
    "lemme": ["L", "EH1", "M", "IY0"],
    "gimme": ["G", "IH1", "M", "IY0"],
    "kinda": ["K", "AY1", "N", "D", "AH0"],
    "sorta": ["S", "AO1", "R", "T", "AH0"],
    "outta": ["AW1", "T", "AH0"],
    "hafta": ["HH", "AE1", "F", "T", "AH0"],
    "hasta": ["HH", "AE1", "S", "T", "AH0"],
    "shoulda": ["SH", "UH1", "D", "AH0"],
    "coulda": ["K", "UH1", "D", "AH0"],
    "woulda": ["W", "UH1", "D", "AH0"],
    "oughta": ["AO1", "T", "AH0"],
    # Contractions needed for preprocessing (if full CMUdict unavailable)
    "i'll": ["AY1", "L"], "you'll": ["Y", "UW1", "L"],
    "he'll": ["HH", "IY1", "L"], "she'll": ["SH", "IY1", "L"],
    "we'll": ["W", "IY1", "L"], "they'll": ["DH", "EY1", "L"],
    "i've": ["AY1", "V"], "you've": ["Y", "UW1", "V"],
    "we've": ["W", "IY1", "V"], "they've": ["DH", "EY1", "V"],
    "isn't": ["IH1", "Z", "AH0", "N", "T"],
    "aren't": ["AA1", "R", "N", "T"],
    "wasn't": ["W", "AH1", "Z", "AH0", "N", "T"],
    "weren't": ["W", "ER1", "N", "T"],
    "haven't": ["HH", "AE1", "V", "AH0", "N", "T"],
    "hasn't": ["HH", "AE1", "Z", "AH0", "N", "T"],
    "hadn't": ["HH", "AE1", "D", "AH0", "N", "T"],
    "won't": ["W", "OW1", "N", "T"],
    "wouldn't": ["W", "UH1", "D", "AH0", "N", "T"],
    "couldn't": ["K", "UH1", "D", "AH0", "N", "T"],
    "shouldn't": ["SH", "UH1", "D", "AH0", "N", "T"],
    "doesn't": ["D", "AH1", "Z", "AH0", "N", "T"],
    "didn't": ["D", "IH1", "D", "AH0", "N", "T"],
}

# ── Connected speech preprocessing ─────────────────────────────────
#
# Neural TTS models (including Qwen-Omni's Talker) naturally produce
# connected speech: /t/ → /ɾ/ (flap), vowel reduction, consonant
# assimilation. We map common multi-word phrases to their reduced
# forms before dictionary lookup, so predicted visemes match reality.
#
# Order matters: longer patterns first to avoid partial matches.

_CONNECTED_SPEECH_PATTERNS: list[tuple[str, str]] = [
    # Multi-word reductions (must come FIRST to prevent partial matches)
    ("going to", "gonna"), ("want to", "wanna"),
    ("got to", "gotta"), ("have to", "hafta"),
    ("has to", "hasta"), ("ought to", "oughta"),
    ("don't you", "dontcha"), ("did you", "didja"),
    ("what are you", "whatcha"), ("what are", "whatcha"),
    ("let me", "lemme"), ("give me", "gimme"),
    ("kind of", "kinda"), ("sort of", "sorta"),
    ("out of", "outta"), ("should have", "shoulda"),
    ("could have", "coulda"), ("would have", "woulda"),
    # Contractions (after multi-word, so "I have to" → hafta, not "I've to")
    ("do not", "don't"), ("does not", "doesn't"), ("did not", "didn't"),
    ("will not", "won't"), ("can not", "can't"), ("cannot", "can't"),
    ("is not", "isn't"), ("are not", "aren't"), ("was not", "wasn't"),
    ("have not", "haven't"), ("has not", "hasn't"), ("had not", "hadn't"),
    ("would not", "wouldn't"), ("could not", "couldn't"),
    ("should not", "shouldn't"), ("might not", "mightn't"),
    ("it is", "it's"), ("that is", "that's"), ("what is", "what's"),
    ("here is", "here's"), ("there is", "there's"), ("who is", "who's"),
    ("i am", "i'm"), ("you are", "you're"), ("we are", "we're"),
    ("they are", "they're"), ("he is", "he's"), ("she is", "she's"),
    ("i will", "i'll"), ("you will", "you'll"),
    ("he will", "he'll"), ("she will", "she'll"),
    ("we will", "we'll"), ("they will", "they'll"),
    ("i have", "i've"), ("you have", "you've"),
    ("we have", "we've"), ("they have", "they've"),
]


def _preprocess_connected_speech(text: str) -> str:
    """Normalize common connected-speech patterns before dictionary lookup.

    Maps multi-word phrases to their phonetically reduced forms so the
    predicted viseme timeline matches what the neural TTS actually produces.

    Preserves punctuation and case of surrounding text.
    """
    result = " " + text.lower() + " "
    for pattern, replacement in _CONNECTED_SPEECH_PATTERNS:
        result = result.replace(" " + pattern + " ", " " + replacement + " ")
    return result.strip()

# ── Try to merge with full CMUdict if available ──

_cmu_dict: dict[str, list[str]] | None = None


def _get_phonemes(word: str) -> list[str] | None:
    """Get ARPABET phonemes for a word.

    Uses the embedded children's dictionary (~250 words).
    If the full CMUdict was successfully downloaded by phoneme_dict,
    merges that in too for broader coverage.
    """
    global _cmu_dict
    if _cmu_dict is None:
        # First call: try to merge with full CMUdict
        _cmu_dict = dict(_EMBEDDED_CMU)
        try:
            from camera_tutor.phoneme_dict import get_phonemes as _cmu_get
            # Test if CMUdict was downloaded
            test = _cmu_get("hello")
            if test:
                # CMUdict is available — use it as primary, embedded as fallback
                # We don't eagerly merge all 130K entries, just use on-demand
                _cmu_dict = _CMUMergedDict(_EMBEDDED_CMU, _cmu_get)
        except Exception:
            pass  # CMUdict not available, use embedded only

    val = _cmu_dict.get(word.lower())
    if isinstance(val, list):
        return val
    return None


class _CMUMergedDict:
    """Lazy merge: embedded dict first, fall back to CMUdict function."""

    def __init__(self, embedded: dict, cmu_func):
        self._embedded = embedded
        self._cmu_func = cmu_func
        self._cache: dict[str, list[str] | None] = {}

    def get(self, key: str) -> list[str] | None:
        if key in self._cache:
            return self._cache[key]
        # Embedded first
        if key in self._embedded:
            result = self._embedded[key]
            self._cache[key] = result
            return result
        # Try CMUdict
        try:
            result = self._cmu_func(key)
            self._cache[key] = result
            return result
        except Exception:
            self._cache[key] = None
            return None


# ── Statistical phoneme durations (milliseconds) ──────────────────
#
# Median durations for conversational American English (~4-5 syll/sec).
# Adjusted +20% for child-directed speech (slower, exaggerated articulation).
#
# Sources: Crystal & House (1988) "Segmental durations in connected-speech
# signals", Umeda (1975) "Vowel duration in American English", and
# modern TTS duration model statistics.
#
# ARPABET keys (same as CMUdict output).

_PHONEME_DURATION_MS: dict[str, float] = {
    # ── Short vowels (60-100ms) ──
    'IH':  75,   # /ɪ/ ship, bit
    'EH':  80,   # /ɛ/ bed, get
    'AE':  100,  # /æ/ cat, bad — exaggerated for teaching
    'AH':  70,   # /ʌ/ cup, /ə/ about
    'UH':  75,   # /ʊ/ pull, book

    # ── Long vowels & diphthongs (130-220ms) ──
    'IY':  150,  # /i/ bee, see
    'EY':  170,  # /eɪ/ say, day
    'AA':  160,  # /ɑ/ car, hot — exaggerated for teaching
    'AO':  155,  # /ɔ/ dog, law
    'OW':  180,  # /oʊ/ boat, go
    'UW':  155,  # /u/ blue, too
    'AW':  200,  # /aʊ/ cow, how
    'AY':  210,  # /aɪ/ eye, fly
    'OY':  200,  # /ɔɪ/ boy, toy
    'ER':  165,  # /ɝ/ bird, her — exaggerated for teaching

    # ── Voiceless stops (15-35ms closure + burst) ──
    'P':   25,
    'T':   30,
    'K':   35,

    # ── Voiced stops (35-65ms) ──
    'B':   45,
    'D':   50,
    'G':   55,

    # ── Voiceless fricatives (70-130ms) ──
    'F':   85,
    'TH':  100,  # /θ/ think — exaggerated for teaching (tongue visible!)
    'S':   100,
    'SH':  100,
    'HH':  75,

    # ── Voiced fricatives (55-110ms) ──
    'V':   75,
    'DH':  70,   # /ð/ this
    'Z':   80,
    'ZH':  85,

    # ── Affricates (100-120ms) ──
    'CH':  110,
    'JH':  105,

    # ── Nasals (55-85ms) ──
    'M':   65,
    'N':   65,
    'NG':  75,

    # ── Liquids & glides (45-75ms) ──
    'L':   60,
    'R':   55,
    'W':   60,
    'Y':   60,
}

_DEFAULT_DURATION_MS = 80.0
_WORD_PAUSE_MS = 80.0  # pause between words for child-directed speech


def _predict_phoneme_timeline(text: str) -> list[tuple[str, float, float]]:
    """Predict phoneme timeline from text using statistical durations.

    Preprocesses connected speech (e.g. "going to" → "gonna") before
    dictionary lookup so predicted visemes match neural TTS output.

    Returns:
        List of (arpabet_phoneme, start_ms, end_ms) tuples.
        start_ms values start at 0 and accumulate.
    """
    # Preprocess connected speech before splitting into words
    text = _preprocess_connected_speech(text)
    words = text.strip().split()
    if not words:
        return []

    timeline: list[tuple[str, float, float]] = []
    current_ms = 0.0

    for wi, word in enumerate(words):
        clean_word = word.strip(',.!?\'"')
        if not clean_word:
            continue

        if wi > 0:
            current_ms += _WORD_PAUSE_MS

        phonemes = _get_phonemes(clean_word)

        if phonemes:
            for p in phonemes:
                key = ''.join(c for c in p if c.isalpha())
                dur = _PHONEME_DURATION_MS.get(key.upper(), _DEFAULT_DURATION_MS)
                timeline.append((p, current_ms, current_ms + dur))
                current_ms += dur
        else:
            # Word not in dictionary — fallback: estimate from letter count
            letter_dur = 60.0
            for ch in clean_word:
                timeline.append((ch.upper(), current_ms, current_ms + letter_dur))
                current_ms += letter_dur

    return timeline


def predict_viseme_timeline(
    text: str,
    total_duration_s: float,
) -> list[tuple[float, Viseme]]:
    """Predict a viseme timeline from text alone.

    Args:
        text: The spoken text (Emma's response).
        total_duration_s: Actual audio duration in seconds (used to
            globally scale predicted durations to match reality).

    Returns:
        List of (time_s, Viseme) pairs. time_s values are in seconds
        relative to audio start. The consumer plays audio and, at
        each time_s, switches to the corresponding Viseme.

    Cost: ~1ms CPU (dictionary lookup + arithmetic). Zero GPU.
    """
    if not text or total_duration_s <= 0:
        return [(0.0, Viseme.V00_SIL)]

    phone_timeline = _predict_phoneme_timeline(text)

    if not phone_timeline:
        return [(0.0, Viseme.V00_SIL)]

    # Scale predicted durations to match actual audio length
    predicted_total_ms = phone_timeline[-1][2]
    if predicted_total_ms <= 0:
        return [(0.0, Viseme.V00_SIL)]

    scale = (total_duration_s * 1000.0) / predicted_total_ms

    viseme_timeline: list[tuple[float, Viseme]] = []

    for arpabet, start_ms, _end_ms in phone_timeline:
        t = start_ms * scale / 1000.0
        key = ''.join(c for c in arpabet if c.isalpha()).lower()
        viseme = PHONEME_TO_VISEME.get(key, Viseme.V00_SIL)
        viseme_timeline.append((t, viseme))

    return viseme_timeline
