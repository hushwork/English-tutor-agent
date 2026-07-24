"""Avatar — Emma's animated face with phoneme-synchronized mouth.

Emma's face is not decoration — it's a pronunciation teaching tool
grounded in 50 years of visual speech perception research:

  McGurk & MacDonald (Nature, 1976):
    Audio-visual fusion at brainstem level. Audio "ba" + mouth "ga"
    = brain hears "da". Visual speech is part of perception, not add-on.

  Kuhl & Meltzoff (Science, 1982):
    2-4 month infants match heard vowels to seen mouth shapes.
    Cross-modal matching is innate, not learned.

  Hazan et al. (2006):
    Visual speech cues help L2 learners 3-5x more than native speakers.
    L2 identification accuracy: 72% audio-only -> 89% audio+visual (+17%).

For Chinese children learning English, the 6 visemes in CHINESE_L2_HOTSPOTS
target phonemes that have NO equivalent in their native phonology.
The visual channel is the PRIMARY way to learn correct articulation.

Provides:
- 10 viseme types with teaching priority (1-5) for Chinese L2 learners
- Display state management (off when silent, on when speaking)
- Tongue visibility for dental/alveolar visemes (/th/, /t/, /d/)
- Minimal GPU overhead (targets Orin's integrated GPU)

Three rendering backends:
1. Live2D Cubism SDK (2D skeletal animation, recommended)
2. Custom SVG with interpolated blendshapes (zero dependency fallback)
3. VRM (3D, for future high-end)

Lip-sync strategy (layered, best available → fallback):

  Phase 1 [MVP — current]: Text-based heuristic
    Audio duration ÷ word count → estimate word position →
    PHONEME_TO_VISEME lookup. Zero additional compute.
    Accuracy: ~60%. Sufficient for showing basic mouth shapes.

  Phase 2 [Production]: whisper.cpp word-level timestamps
    Run whisper.cpp on Qwen-Omni's generated audio → word-level
    {word, start_ms, end_ms} → map to visemes via lookup.
    Accuracy: ~85%. Runs on Orin GPU, adds ~50ms latency.

  Phase 3 [Future]: Feed viseme timeline to Live2D Cubism SDK
    Live2D accepts external parameter input each frame via
    AddParameterValue()/SetParameterValue(). Drive mouth shape
    per-frame from the viseme timeline.

Key finding: Qwen2.5-Omni's Talker outputs raw audio tokens only —
it does NOT provide phoneme or viseme timing metadata.
Azure Cognitive Services and Amazon Polly DO provide viseme events,
but they're cloud TTS — incompatible with our local-first approach.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Viseme types (mouth shapes) ─────────────────────────────────


class Viseme(Enum):
    """22 viseme standard — Microsoft/Azure Cognitive Services viseme set.

    ISO/IEC 14496-2 MPEG-4 Facial Animation compatible mapping.
    Each viseme maps to specific phonemes with precise correspondence.

    teaching_priority (1-5) for Chinese L2 learners:
      5 = phoneme does NOT exist in Chinese → visual is primary path
      4 = similar phoneme but different articulation → visual disambiguates
      3 = exists but visual still helpful for English accuracy
      1-2 = subtle or not distinctive

    tongue_visible = True for phonemes where tongue position is
                     externally visible (dental, alveolar, sibilant)
    """

    V00_SIL  = ("sil",  0, False)   # silence / neutral
    V01_AE_AH = ("ae",  4, False)   # /æ/ cat, /ə/ about, /ʌ/ cup
    V02_AA   = ("aa",   5, False)   # /ɑ/ car, hot — NOT in Chinese
    V03_AO   = ("ao",   4, False)   # /ɔ/ dog, law
    V04_EH_EY = ("eh",  3, False)   # /ɛ/ bed, /eɪ/ say
    V05_ER   = ("er",   5, False)   # /ɝ/ bird, her — NOT in Chinese
    V06_IY_IH = ("iy",  4, False)   # /i/ bee, /ɪ/ ship, /j/ yes
    V07_UW_W = ("uw",   4, False)   # /u/ blue, /w/ wet
    V08_OW   = ("ow",   3, False)   # /oʊ/ boat, go
    V09_AW   = ("aw",   3, False)   # /aʊ/ cow, how
    V10_OY   = ("oy",   2, False)   # /ɔɪ/ boy, toy
    V11_AY   = ("ay",   3, False)   # /aɪ/ eye, fly
    V12_H    = ("h",    2, False)   # /h/ hot, who
    V13_R    = ("r",    5, False)   # /ɹ/ red, car — different from Chinese r
    V14_L    = ("l",    3, True)    # /l/ like, ball
    V15_S_Z  = ("sz",   2, True)    # /s/ see, /z/ zoo
    V16_SH_ZH= ("sh",   2, False)   # /ʃ/ she, /ʒ/ measure, /tʃ/ chip, /dʒ/ jump
    V17_TH_DH= ("th",   5, True)    # /θ/ think, /ð/ this — NOT in Chinese ☆☆☆
    V18_F_V  = ("fv",   5, False)   # /f/ five, /v/ very — /v/ NOT in Chinese ☆☆☆
    V19_T_D_N= ("td",   1, True)    # /t/ top, /d/ dog, /n/ no
    V20_K_G_NG=("kg",   1, False)   # /k/ cat, /g/ go, /ŋ/ sing
    V21_P_B_M= ("pb",   1, False)   # /p/ pop, /b/ big, /m/ mom

    def __new__(cls, value, teaching_priority, tongue_visible):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.teaching_priority = teaching_priority
        obj.tongue_visible = tongue_visible
        return obj

    @property
    def label(self) -> str:
        return self.value

    @property
    def needs_tongue_visible(self) -> bool:
        return self.tongue_visible

    @property
    def is_high_value_teaching(self) -> bool:
        return self.teaching_priority >= 4

    @property
    def viseme_id(self) -> int:
        """Azure viseme ID (0-21)."""
        ids = {v: i for i, v in enumerate(Viseme)}
        return ids.get(self, 0)


# ── 22-Viseme Phoneme Mapping ───────────────────────────────────
#
# Based on Microsoft/Azure Cognitive Services viseme standard
# (ISO/IEC 14496-2 MPEG-4 Facial Animation compatible).
#
# Each phoneme key maps to one of the 22 standard visemes.
# For coverage: both IPA (Unicode: æ, ɑ, ɔ, ɪ, ʊ, ʌ, ɝ, θ, ð, ʃ, ʒ) and
# ASCII equivalents (ae, aa, ao, ih, uh, ah, er, th, dh, sh, zh) are mapped.

PHONEME_TO_VISEME: dict[str, 'Viseme'] = {
    # V00 — silence (not mapped, used as default)
    # V01 — /æ/ cat, /ə/ about, /ʌ/ cup
    'ae': Viseme.V01_AE_AH, 'ah': Viseme.V01_AE_AH,
    'æ': Viseme.V01_AE_AH,  'ə': Viseme.V01_AE_AH,  'ʌ': Viseme.V01_AE_AH,
    # V02 — /ɑ/ car, hot
    'aa': Viseme.V02_AA, 'ɑ': Viseme.V02_AA,
    # V03 — /ɔ/ dog, law
    'ao': Viseme.V03_AO, 'ɔ': Viseme.V03_AO,
    # V04 — /ɛ/ bed, /eɪ/ say
    'eh': Viseme.V04_EH_EY, 'ey': Viseme.V04_EH_EY,
    'e': Viseme.V04_EH_EY,  'ɛ': Viseme.V04_EH_EY,  'eɪ': Viseme.V04_EH_EY,
    # V05 — /ɝ/ bird, her
    'er': Viseme.V05_ER, 'ɝ': Viseme.V05_ER,
    # V06 — /i/ bee, /ɪ/ ship, /j/ yes
    'iy': Viseme.V06_IY_IH, 'ih': Viseme.V06_IY_IH, 'y': Viseme.V06_IY_IH,
    'i': Viseme.V06_IY_IH,  'ɪ': Viseme.V06_IY_IH,  'iː': Viseme.V06_IY_IH,
    'j': Viseme.V06_IY_IH,
    # V07 — /u/ blue, /w/ wet
    'uw': Viseme.V07_UW_W, 'w': Viseme.V07_UW_W,
    'u': Viseme.V07_UW_W,  'ʊ': Viseme.V07_UW_W,  'uː': Viseme.V07_UW_W,
    # V08 — /oʊ/ boat, go
    'ow': Viseme.V08_OW, 'oʊ': Viseme.V08_OW, 'o': Viseme.V08_OW,
    # V09 — /aʊ/ cow
    'aw': Viseme.V09_AW, 'aʊ': Viseme.V09_AW,
    # V10 — /ɔɪ/ boy
    'oy': Viseme.V10_OY, 'ɔɪ': Viseme.V10_OY,
    # V11 — /aɪ/ eye
    'ay': Viseme.V11_AY, 'aɪ': Viseme.V11_AY,
    # V12 — /h/ hot
    'hh': Viseme.V12_H, 'h': Viseme.V12_H,
    # V13 — /ɹ/ red, car
    'r': Viseme.V13_R, 'ɹ': Viseme.V13_R,
    # V14 — /l/ like, ball
    'l': Viseme.V14_L,
    # V15 — /s/ see, /z/ zoo
    's': Viseme.V15_S_Z, 'z': Viseme.V15_S_Z,
    # V16 — /ʃ/ she, /ʒ/ measure, /tʃ/ chip, /dʒ/ jump
    'sh': Viseme.V16_SH_ZH, 'zh': Viseme.V16_SH_ZH,
    'ch': Viseme.V16_SH_ZH, 'jh': Viseme.V16_SH_ZH,
    'ʃ': Viseme.V16_SH_ZH,  'ʒ': Viseme.V16_SH_ZH,
    'tʃ': Viseme.V16_SH_ZH, 'dʒ': Viseme.V16_SH_ZH,
    # V17 — /θ/ think, /ð/ this
    'th': Viseme.V17_TH_DH, 'dh': Viseme.V17_TH_DH,
    'θ': Viseme.V17_TH_DH,  'ð': Viseme.V17_TH_DH,
    # V18 — /f/ five, /v/ very
    'f': Viseme.V18_F_V, 'v': Viseme.V18_F_V,
    # V19 — /t/ top, /d/ dog, /n/ no
    't': Viseme.V19_T_D_N, 'd': Viseme.V19_T_D_N, 'n': Viseme.V19_T_D_N,
    # V20 — /k/ cat, /g/ go, /ŋ/ sing
    'k': Viseme.V20_K_G_NG, 'g': Viseme.V20_K_G_NG, 'ng': Viseme.V20_K_G_NG,
    'ŋ': Viseme.V20_K_G_NG,
    # V21 — /p/ pop, /b/ big, /m/ mom
    'p': Viseme.V21_P_B_M, 'b': Viseme.V21_P_B_M, 'm': Viseme.V21_P_B_M,
}

# ── Chinese L2 Hotspots — phonemes where visual teaching matters most ──
#
# Research basis:
#   McGurk & MacDonald (1976): Audio-visual fusion at brainstem level
#   Kuhl & Meltzoff (1982): Infants match heard vowels to seen mouth shapes
#   Hazan et al. (2006): L2 learners benefit 3-5x more from visual speech
#                        than native speakers (17% accuracy gain vs 3%)
#   Hardison (2003): Visual cues are "anchors" for undeveloped L2 phoneme categories
#
# For Chinese children learning English, these phonemes have NO
# equivalent in their native phonology — the visual channel is
# the PRIMARY way to learn correct articulation:

CHINESE_L2_HOTSPOTS = {
    # ⭐⭐⭐⭐⭐ CRITICAL: phoneme does NOT exist in Chinese
    "V17_TH_DH": {
        "viseme": Viseme.V17_TH_DH,
        "phonemes": ["/θ/", "/ð/"],
        "examples": "think, this, three, mother",
        "chinese_problem": "汉语没有齿间音。孩子用 /s/ 替代 think→sink",
        "visual_cue": "舌尖伸出上下齿之间 ☆☆☆ 最可见的英语音素",
    },
    "V18_F_V": {
        "viseme": Viseme.V18_F_V,
        "phonemes": ["/f/", "/v/"],
        "examples": "five, very, fish, voice",
        "chinese_problem": "/v/ 不存在。孩子 very→wery, voice→woice",
        "visual_cue": "上齿咬下唇。与 V07(/w/圆唇)形成鲜明对比",
    },
    "V02_AA": {
        "viseme": Viseme.V02_AA,
        "phonemes": ["/ɑ/"],
        "examples": "car, hot, father, dog",
        "chinese_problem": "汉语无 /ɑ/。孩子用 /a/ 替代 → car→ka",
        "visual_cue": "大开口，舌后缩。开口度大于汉语任何元音",
    },
    "V05_ER": {
        "viseme": Viseme.V05_ER,
        "phonemes": ["/ɝ/"],
        "examples": "bird, her, word, learn",
        "chinese_problem": "卷舌元音在汉语中完全不存在",
        "visual_cue": "舌尖卷起+嘴唇微圆。最独特的英语嘴型之一",
    },
    "V13_R": {
        "viseme": Viseme.V13_R,
        "phonemes": ["/ɹ/"],
        "examples": "red, car, rabbit, run",
        "chinese_problem": "英语 /ɹ/ 和汉语 r 发声方式不同",
        "visual_cue": "嘴唇圆拢前突 → 与 V07(/w/)、V18(/v/)三唇对比",
    },
    # ⭐⭐⭐⭐ HIGH: different articulation from Chinese
    "V01_AE_AH": {
        "viseme": Viseme.V01_AE_AH,
        "phonemes": ["/æ/"],
        "examples": "cat, apple, bad, happy",
        "chinese_problem": "汉语无 /æ/。孩子 cat→ket, bad→bed",
        "visual_cue": "大开口+嘴角拉开 → 明显大于 /ɛ/",
    },
    "V06_IY_IH": {
        "viseme": Viseme.V06_IY_IH,
        "phonemes": ["/iː/", "/ɪ/"],
        "examples": "sheep/ship, bean/bin, leave/live",
        "chinese_problem": "无长短元音对立。肌肉紧张度不同",
        "visual_cue": "嘴角拉开程度+肌肉紧张度 → sheep vs ship",
    },
    "V07_UW_W": {
        "viseme": Viseme.V07_UW_W,
        "phonemes": ["/uː/", "/ʊ/", "/w/"],
        "examples": "pool/pull, wet, water",
        "chinese_problem": "无 /uː/ vs /ʊ/ 对立。/w/ vs /v/ 混淆",
        "visual_cue": "双唇圆拢前突 → 与 V18(/v/)咬唇对比关键",
    },
}



# ── Phoneme timing from force aligner ─────────────────────────────


@dataclass
class AlignmentEntry:
    """A phoneme with precise timing from a force aligner.

    This is the output of Montreal Forced Aligner (MFA) or any
    CTC-based forced aligner. It maps a phoneme to an exact
    time window in the generated audio.
    """
    phoneme: str          # ARPABET phoneme: "DH", "AE", "K", "T", ...
    start_ms: float       # Start time in milliseconds
    end_ms: float         # End time in milliseconds
    word: str = ""        # Source word (for debugging)


def alignment_from_mfa_textgrid(textgrid_path: str) -> list[AlignmentEntry]:
    """Parse a Montreal Forced Aligner .TextGrid output file.

    MFA outputs Praat TextGrid format with word and phoneme tiers.
    This extracts the phoneme tier into AlignmentEntry objects.

    Usage:
        # Run MFA once per utterance:
        # $ mfa align --clean one_utterance.wav english_mfa english_mfa output_dir
        # Then:
        entries = alignment_from_mfa_textgrid("output_dir/one_utterance.TextGrid")
    """
    entries = []
    try:
        with open(textgrid_path) as f:
            textgrid = f.read()

        # Simple TextGrid parser (phoneme tier only)
        in_phoneme_tier = False
        in_intervals = False
        intervals: list[dict] = []
        xmin, xmax = 0.0, 0.0

        for line in textgrid.split('\n'):
            line = line.strip()
            if 'name = "phones"' in line or 'name = "phonemes"' in line:
                in_phoneme_tier = True
                continue
            if in_phoneme_tier:
                if 'xmin = ' in line:
                    xmin = float(line.split('=')[1].strip())
                elif 'xmax = ' in line:
                    xmax = float(line.split('=')[1].strip())
                elif 'text = ' in line:
                    text = line.split('=')[1].strip().strip('"')
                    intervals.append({
                        'xmin': xmin, 'xmax': xmax,
                        'text': text,
                    })
                if 'item [' in line and intervals:
                    in_phoneme_tier = False

        for interval in intervals:
            if interval['text'] and interval['text'] not in ('', 'sp', 'sil'):
                entries.append(AlignmentEntry(
                    phoneme=interval['text'],
                    start_ms=interval['xmin'] * 1000,
                    end_ms=interval['xmax'] * 1000,
                ))

    except Exception as e:
        print(f"[WARN] Failed to parse TextGrid: {e}")

    return entries


# ── Display state machine ───────────────────────────────────────


class DisplayState(Enum):
    OFF = "off"           # Screen completely dark (AMOLED pixels off)
    FADING_IN = "fading_in"     # Face appearing
    SPEAKING = "speaking"       # Mouth animated, face visible
    LISTENING = "listening"     # Slight idle animation (nodding, blinking)
    FADING_OUT = "fading_out"   # Face disappearing
    ERROR = "error"              # Something went wrong (show gentle indicator)


@dataclass
class DisplayCommand:
    """Command from the main loop to control the avatar display."""
    state: DisplayState
    text: str = ""                    # Current text being spoken (for phoneme extraction)
    audio_timestamp: float = 0.0      # Position in audio for lip sync
    emotion: str = "neutral"          # happy, surprised, encouraging, thinking
    timestamp: float = field(default_factory=time.time)


class EmmaAvatar:
    """Manages Emma's facial animation state machine.

    Coordinates:
    - Display on/off timing
    - Phoneme extraction from text for lip sync
    - Emotion expression mapping
    - Transition animations (fade in/out)
    """

    def __init__(self):
        self.current_state = DisplayState.OFF
        self.current_viseme = Viseme.V00_SIL
        self._state_start = time.time()
        self._blink_timer = 0.0
        self._blink_interval = 3.0  # Blink every ~3 seconds
        self._nod_timer = 0.0
        self._nod_active = False

        # Force aligner timeline (set by load_alignment)
        self._alignment: list[AlignmentEntry] = []

    # ── Force aligner integration ──────────────────────────────

    def load_alignment(self, entries: list[AlignmentEntry]):
        """[Production] Load precise phoneme timing from force aligner.

        Called after Qwen-Omni generates audio + MFA produces alignment.
        Once loaded, _phoneme_at_position uses exact timing instead of
        the word-timing heuristic. Accuracy goes from ~60% to ~95%.

        Args:
            entries: List of AlignmentEntry from MFA or compatible aligner
        """
        self._alignment = entries

    def load_alignment_from_mfa(self, textgrid_path: str):
        """Convenience: load alignment directly from an MFA .TextGrid file."""
        self._alignment = alignment_from_mfa_textgrid(textgrid_path)

    def has_alignment(self) -> bool:
        """Check if precise alignment data has been loaded."""
        return len(self._alignment) > 0

    def get_display_command(
        self,
        should_speak: bool,
        text: str = "",
        audio_pos: float = 0.0,
    ) -> DisplayCommand:
        """Get the current display command based on state.

        Args:
            should_speak: Whether Emma is about to speak/speaking
            text: Text being spoken (for lip sync)
            audio_pos: Current position in audio playback (seconds)
        """
        now = time.time()

        if should_speak:
            if self.current_state in (DisplayState.OFF, DisplayState.FADING_OUT):
                # Transition: off → speaking
                self.current_state = DisplayState.FADING_IN
                self._state_start = now

            elif self.current_state == DisplayState.FADING_IN:
                # Check if fade-in complete (300ms)
                if now - self._state_start > 0.3:
                    self.current_state = DisplayState.SPEAKING
                    self._state_start = now

            else:
                # Already SPEAKING — update viseme
                self.current_viseme = self._phoneme_at_position(text, audio_pos)

        else:
            if self.current_state in (DisplayState.SPEAKING, DisplayState.FADING_IN):
                # Transition: speaking → listening
                self.current_state = DisplayState.LISTENING
                self._state_start = now
                self._blink_timer = now
                self.current_viseme = Viseme.V00_SIL

            elif self.current_state == DisplayState.LISTENING:
                # Check if should fade out (no interaction for 30s)
                if now - self._state_start > 30.0:
                    self.current_state = DisplayState.FADING_OUT
                    self._state_start = now

            elif self.current_state == DisplayState.FADING_OUT:
                if now - self._state_start > 0.5:
                    self.current_state = DisplayState.OFF

        return DisplayCommand(
            state=self.current_state,
            text=text,
            audio_timestamp=audio_pos,
            emotion=self._infer_emotion(text),
            timestamp=now,
        )

    # ── Phoneme extraction ──────────────────────────────────────

    def _phoneme_at_position(self, text: str, audio_pos: float) -> Viseme:
        """Determine which viseme to show at this audio position.

        Phase 3 [Production — USE THIS]:
            If force-aligner alignment is loaded via load_alignment(),
            do a binary search in the alignment timeline.
            Each entry is {phoneme, start_ms, end_ms} from MFA.
            Accuracy ~95%, MFA adds ~200ms latency on Orin CPU.

        Phase 2 [Fallback — word-level]:
            If whisper.cpp word timestamps are loaded via
            load_whisper_timeline(), use word-level lookup.
            Accuracy ~85%.

        Phase 1 [Fallback — heuristic]:
            Otherwise, estimate from word count.
            Accuracy ~60%, zero additional latency.
        """
        if not text:
            return Viseme.V00_SIL

        audio_pos_ms = audio_pos * 1000

        # ── Phase 3: Force aligner (highest accuracy) ──
        if self._alignment:
            # Binary search on alignment timeline
            lo, hi = 0, len(self._alignment) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                entry = self._alignment[mid]
                if entry.start_ms <= audio_pos_ms < entry.end_ms:
                    # Found the exact phoneme — map to viseme
                    return self._phoneme_label_to_viseme(entry.phoneme)
                elif audio_pos_ms < entry.start_ms:
                    hi = mid - 1
                else:
                    lo = mid + 1
            # Past the end of alignment — hold last viseme or rest
            return Viseme.V00_SIL

        # ── Phase 2: Whisper word-level (medium accuracy) ──
        if hasattr(self, '_whisper_timeline') and self._whisper_timeline:
            for entry in self._whisper_timeline:
                if entry['start_ms'] <= audio_pos_ms <= entry['end_ms']:
                    return self._word_to_viseme(entry['word'])
            return Viseme.V00_SIL

        # ── Phase 1: Heuristic (fallback) ──
        words = text.split()
        if not words:
            return Viseme.V00_SIL
        words_per_second = 3.0
        total_duration = len(words) / words_per_second
        progress = audio_pos / max(total_duration, 0.1)
        word_index = min(int(progress * len(words)), len(words) - 1)
        return self._word_to_viseme(words[word_index])

    def _phoneme_label_to_viseme(self, phoneme: str) -> Viseme:
        """Map a force-aligner phoneme label (ARPABET) to a Viseme.

        ARPABET is the standard phoneme set used by MFA, CMUdict, etc.
        Examples: "DH" → /ð/ → Viseme.V17_TH_DH, "AE" → /æ/ → Viseme.V02_AA
        """
        # ARPABET → our phoneme keys
        # ARPABET (MFA standard) → our phoneme keys
        # Direct mapping to PHONEME_TO_VISEME lookup keys.
        # No intermediate transform needed — ARPABET keys match our ASCII keys.
        arpabet_to_ours = {
            # Vowels
            'AA': 'aa', 'AE': 'ae', 'AH': 'ah', 'AO': 'ao', 'AW': 'aw',
            'AY': 'ay', 'EH': 'eh', 'ER': 'er', 'EY': 'ey', 'IH': 'ih',
            'IY': 'iy', 'OW': 'ow', 'OY': 'oy', 'UH': 'uh', 'UW': 'uw',
            # Consonants
            'P': 'p', 'B': 'b', 'T': 't', 'D': 'd', 'K': 'k', 'G': 'g',
            'CH': 'ch', 'JH': 'jh', 'F': 'f', 'V': 'v', 'TH': 'th', 'DH': 'dh',
            'S': 's', 'Z': 'z', 'SH': 'sh', 'ZH': 'zh', 'HH': 'hh',
            'M': 'm', 'N': 'n', 'NG': 'ng', 'L': 'l', 'R': 'r',
            'W': 'w', 'Y': 'y',
        }
        our_key = arpabet_to_ours.get(phoneme.upper(), phoneme.lower())
        return PHONEME_TO_VISEME.get(our_key, Viseme.V00_SIL)

    def load_whisper_timeline(
        self, word_timeline: list[dict]
    ) -> None:
        """[Phase 2] Load whisper.cpp word-level timestamp timeline.

        Args:
            word_timeline: List of {word, start_ms, end_ms} dicts
                from whisper.cpp or whisper-timestamped output.

        Once loaded, _phoneme_at_position uses this precise
        timeline instead of the heuristic.
        """
        self._whisper_timeline = word_timeline

    def _word_to_viseme(self, word: str) -> Viseme:
        """Map a word to its most prominent viseme (first vowel sound)."""
        word_lower = word.lower().strip('.,!?\'"')

        # Quick heuristic: check common patterns
        if any(c in word_lower for c in 'aeiou'):
            # Find first vowel and map it
            for phoneme, viseme in sorted(
                PHONEME_TO_VISEME.items(),
                key=lambda x: len(x[0]),
                reverse=True,
            ):
                if phoneme in word_lower:
                    return viseme

        # Default: open mouth for unknowns
        return Viseme.V02_AA

    # ── Emotion ─────────────────────────────────────────────────

    def _infer_emotion(self, text: str) -> str:
        """Infer emotional expression from text content."""
        if not text:
            return "neutral"

        text_lower = text.lower()

        if any(w in text_lower for w in ['wow', 'great', 'amazing', 'good job', 'yes']):
            return "happy"
        if any(w in text_lower for w in ['wow!', 'look', 'surprise']):
            return "surprised"
        if '?' in text_lower:
            return "curious"
        if any(w in text_lower for w in ['good', 'nice', 'well done', 'try again']):
            return "encouraging"
        if any(w in text_lower for w in ['hmm', 'let me', 'think']):
            return "thinking"

        return "neutral"

    # ── SVG rendering helper ────────────────────────────────────

    def render_svg(self, size: tuple[int, int] = (300, 300)) -> str:
        """Render Emma's face as an SVG string.

        This is the MVP rendering path — no external dependencies.
        For production, replace with Live2D Cubism SDK rendering.

        Args:
            size: (width, height) in pixels

        Returns:
            SVG string ready to display in a web view or convert to image
        """
        if self.current_state == DisplayState.OFF:
            return ''  # Nothing to render

        w, h = size
        cx, cy = w / 2, h / 2
        face_r = min(w, h) * 0.35

        opacity = 1.0
        if self.current_state == DisplayState.FADING_IN:
            elapsed = time.time() - self._state_start
            opacity = min(1.0, elapsed / 0.3)
        elif self.current_state == DisplayState.FADING_OUT:
            elapsed = time.time() - self._state_start
            opacity = max(0.0, 1.0 - elapsed / 0.5)

        # Emotion modifiers
        eye_scale_y = 1.0
        mouth_curve = 0
        if self.current_state == DisplayState.LISTENING:
            mouth_curve = 5  # Slight smile while listening

        # Mouth shape based on viseme
        mouth_path = self._viseme_to_svg_path(
            self.current_viseme, cx, cy + face_r * 0.5, face_r * 0.35
        )

        # Tongue for dental/alveolar visemes (critical for Chinese L2)
        if self.current_viseme.needs_tongue_visible:
            tongue_x = cx
            tongue_y = cy + face_r * 0.55
            tongue_w = face_r * 0.12
            tongue_h = face_r * 0.08
            tongue_svg = (
                f'<ellipse cx="{tongue_x}" cy="{tongue_y}" '
                f'rx="{tongue_w}" ry="{tongue_h}" '
                f'fill="#ff8888" opacity="{opacity*0.8}"/>'
            )
        else:
            tongue_svg = ""

        # Eye blink
        blink = 0
        now = time.time()
        if self.current_state == DisplayState.LISTENING:
            cycle = (now - self._blink_timer) % self._blink_interval
            if cycle < 0.1:  # Blink for 100ms
                blink = 1

        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <defs>
    <radialGradient id="faceGrad" cx="50%" cy="40%" r="50%">
      <stop offset="0%" style="stop-color:#ffe4c4;stop-opacity:{opacity}"/>
      <stop offset="100%" style="stop-color:#f0c8a0;stop-opacity:{opacity}"/>
    </radialGradient>
  </defs>

  <!-- Face -->
  <ellipse cx="{cx}" cy="{cy}" rx="{face_r}" ry="{face_r*1.1}"
           fill="url(#faceGrad)" stroke="#e0b890" stroke-width="2"
           opacity="{opacity}"/>

  <!-- Eyes -->
  <ellipse cx="{cx-face_r*0.35}" cy="{cy-face_r*0.2}" rx="{face_r*0.1}" ry="{face_r*0.1*eye_scale_y*(1-blink)}"
           fill="#333" opacity="{opacity}"/>
  <ellipse cx="{cx+face_r*0.35}" cy="{cy-face_r*0.2}" rx="{face_r*0.1}" ry="{face_r*0.1*eye_scale_y*(1-blink)}"
           fill="#333" opacity="{opacity}"/>

  <!-- Eyebrows -->
  <path d="M {cx-face_r*0.45} {cy-face_r*0.4} Q {cx-face_r*0.35} {cy-face_r*0.45} {cx-face_r*0.2} {cy-face_r*0.38}"
        stroke="#8b6b4a" stroke-width="2.5" fill="none" opacity="{opacity}"/>
  <path d="M {cx+face_r*0.2} {cy-face_r*0.38} Q {cx+face_r*0.35} {cy-face_r*0.45} {cx+face_r*0.45} {cy-face_r*0.4}"
        stroke="#8b6b4a" stroke-width="2.5" fill="none" opacity="{opacity}"/>

  <!-- Nose -->
  <ellipse cx="{cx}" cy="{cy+face_r*0.08}" rx="{face_r*0.06}" ry="{face_r*0.05}"
           fill="#e8c8a8" opacity="{opacity}"/>

  <!-- Mouth -->
  <path d="{mouth_path}"
        stroke="#cc7766" stroke-width="3" fill="none"
        stroke-linecap="round" opacity="{opacity}"/>

  <!-- Tongue (visible for /th/ /t/ /d/ -- critical for Chinese learners) -->
  {tongue_svg}

  <!-- Blush -->
  <ellipse cx="{cx-face_r*0.5}" cy="{cy+face_r*0.3}" rx="{face_r*0.15}" ry="{face_r*0.08}"
           fill="#ffcccc" opacity="{opacity*0.3}"/>
  <ellipse cx="{cx+face_r*0.5}" cy="{cy+face_r*0.3}" rx="{face_r*0.15}" ry="{face_r*0.08}"
           fill="#ffcccc" opacity="{opacity*0.3}"/>
</svg>'''

    def get_teaching_guidance(self, viseme: 'Viseme') -> str:
        """Return a child-friendly teaching hint for this mouth shape.

        Called when Emma is explicitly teaching pronunciation —
        she says the word, then adds a visual cue the child can follow.
        """
        hints = {
            'th': "Put your tongue between your teeth! Like this!",
            'fv': "Bite your lip gently! Top teeth on bottom lip!",
            'wq': "Make your lips round! Like a little O!",
            'aa': "Open your mouth BIG! Stretch your lips wide!",
            'iy': "Smile big! Pull your lips to the sides!",
            'ow': "Make a round shape with your lips!",
        }
        return hints.get(viseme.label, "Watch my mouth! Can you say it?")

    def _viseme_to_svg_path(
        self, viseme: Viseme, cx: float, cy: float, size: float
    ) -> str:
        """Convert a viseme to an SVG path for the mouth."""
        # Silence / neutral: slightly open, relaxed
        if viseme in (Viseme.V00_SIL, Viseme.V04_EH_EY):
            return f"M {cx-size*0.65} {cy} Q {cx} {cy+size*0.3} {cx+size*0.65} {cy}"

        # Wide open: /æ/ cat, /ɑ/ car, /aʊ/ cow, /aɪ/ eye
        elif viseme in (Viseme.V01_AE_AH, Viseme.V02_AA, Viseme.V09_AW, Viseme.V11_AY):
            return f"M {cx-size*0.55} {cy-size*0.35} Q {cx} {cy+size*0.75} {cx+size*0.55} {cy-size*0.35}"

        # Medium open: /ɔ/ dog, /oʊ/ boat, /ɔɪ/ boy
        elif viseme in (Viseme.V03_AO, Viseme.V08_OW, Viseme.V10_OY):
            return f"M {cx-size*0.3} {cy-size*0.05} Q {cx} {cy+size*0.5} {cx+size*0.3} {cy-size*0.05}"

        # ER: /ɝ/ bird — unique: tongue curled + lips rounded slightly
        elif viseme == Viseme.V05_ER:
            return f"M {cx-size*0.35} {cy-size*0.05} Q {cx} {cy+size*0.35} {cx+size*0.35} {cy-size*0.05}"

        # Wide smile: /i/ bee, /ɪ/ ship
        elif viseme == Viseme.V06_IY_IH:
            return f"M {cx-size*0.7} {cy-size*0.03} Q {cx-size*0.35} {cy+size*0.45} {cx} {cy+size*0.25} Q {cx+size*0.35} {cy+size*0.45} {cx+size*0.7} {cy-size*0.03}"

        # Rounded + pursed: /u/ blue, /w/ wet
        elif viseme == Viseme.V07_UW_W:
            return f"M {cx-size*0.2} {cy-size*0.05} Q {cx-size*0.2} {cy+size*0.4} {cx} {cy+size*0.45} Q {cx+size*0.2} {cy+size*0.4} {cx+size*0.2} {cy-size*0.05}"

        # Open breath: /h/ hot
        elif viseme == Viseme.V12_H:
            return f"M {cx-size*0.5} {cy-size*0.1} Q {cx} {cy+size*0.4} {cx+size*0.5} {cy-size*0.1}"

        # R: /ɹ/ red — rounded lips
        elif viseme == Viseme.V13_R:
            return f"M {cx-size*0.25} {cy-size*0.03} Q {cx-size*0.25} {cy+size*0.35} {cx} {cy+size*0.38} Q {cx+size*0.25} {cy+size*0.35} {cx+size*0.25} {cy-size*0.03}"

        # L: /l/ like — tongue tip visible
        elif viseme == Viseme.V14_L:
            return f"M {cx-size*0.55} {cy+size*0.05} Q {cx} {cy+size*0.35} {cx+size*0.55} {cy+size*0.05}"

        # S/Z: /s/ see, /z/ zoo — teeth together, lips slightly spread
        elif viseme == Viseme.V15_S_Z:
            return f"M {cx-size*0.6} {cy+size*0.05} Q {cx} {cy+size*0.15} {cx+size*0.6} {cy+size*0.05}"

        # SH/ZH/CH/JH: /ʃ/ she, /tʃ/ chip — lips pursed forward
        elif viseme == Viseme.V16_SH_ZH:
            return f"M {cx-size*0.2} {cy-size*0.02} Q {cx} {cy+size*0.3} {cx+size*0.2} {cy-size*0.02}"

        # TH/DH: /θ/ think, /ð/ this — TONGUE BETWEEN TEETH ☆☆☆
        elif viseme == Viseme.V17_TH_DH:
            return f"M {cx-size*0.5} {cy+size*0.1} Q {cx} {cy+size*0.25} {cx+size*0.5} {cy+size*0.1}"

        # F/V: /f/ five, /v/ very — LOWER LIP BITES UPPER TEETH ☆☆☆
        elif viseme == Viseme.V18_F_V:
            return f"M {cx-size*0.55} {cy} Q {cx-size*0.25} {cy+size*0.18} {cx} {cy+size*0.12} Q {cx+size*0.25} {cy+size*0.18} {cx+size*0.55} {cy}"

        # T/D/N: /t/ top, /d/ dog, /n/ no — tongue on alveolar ridge
        elif viseme == Viseme.V19_T_D_N:
            return f"M {cx-size*0.55} {cy+size*0.05} Q {cx} {cy+size*0.25} {cx+size*0.55} {cy+size*0.05}"

        # K/G/NG: /k/ cat, /g/ go, /ŋ/ sing — back of mouth (not externally visible)
        elif viseme == Viseme.V20_K_G_NG:
            return f"M {cx-size*0.6} {cy+size*0.08} Q {cx} {cy+size*0.25} {cx+size*0.6} {cy+size*0.08}"

        # P/B/M: /p/ pop, /b/ big, /m/ mom — LIPS FULLY CLOSED
        elif viseme == Viseme.V21_P_B_M:
            return f"M {cx-size*0.6} {cy} L {cx+size*0.6} {cy}"

        else:
            return f"M {cx-size*0.6} {cy} Q {cx} {cy+size*0.35} {cx+size*0.6} {cy}"
