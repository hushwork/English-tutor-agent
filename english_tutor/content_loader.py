"""Daily reading content — RSS feed fetching, article extraction, and difficulty grading."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
import httpx
from readability import Document as ReadabilityDoc

# ── Default RSS sources (tiered by difficulty) ──────────────────────

RSS_SOURCES: dict[str, dict[str, str | list[str]]] = {
    "guardian_world": {
        "url": "https://www.theguardian.com/world/rss",
        "label": "The Guardian (World)",
        "level": "intermediate",
        "tags": ["news", "world"],
    },
    "npr_news": {
        "url": "https://feeds.npr.org/1001/rss.xml",
        "label": "NPR News",
        "level": "intermediate",
        "tags": ["news", "us"],
    },
    "nytimes": {
        "url": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "label": "New York Times",
        "level": "advanced",
        "tags": ["news", "top"],
    },
    "guardian_tech": {
        "url": "https://www.theguardian.com/technology/rss",
        "label": "The Guardian (Tech)",
        "level": "intermediate",
        "tags": ["tech"],
    },
    "tech_crunch": {
        "url": "https://techcrunch.com/feed/",
        "label": "TechCrunch",
        "level": "advanced",
        "tags": ["tech", "startup"],
    },
}


@dataclass
class Article:
    """A single article with extracted content and metadata."""

    title: str
    url: str
    source: str
    summary: str = ""
    content: str = ""
    text: str = ""
    word_count: int = 0
    difficulty: str = "unknown"  # beginner / intermediate / advanced
    new_words: list[dict] = field(default_factory=list)
    fetched_at: str = ""
    language: str = "en"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Article:
        return cls(**d)


# ── Feed Fetcher ────────────────────────────────────────────────────

class FeedFetcher:
    """Fetch and extract articles from RSS feeds."""

    def __init__(
        self,
        sources: dict | None = None,
        cache_dir: str | Path | None = None,
        max_articles_per_feed: int = 5,
    ):
        self.sources = sources or RSS_SOURCES
        base = os.environ.get(
            "ENGLISH_TUTOR_DATA_DIR",
            str(Path(__file__).resolve().parent.parent / ".english-tutor-data"),
        )
        self.cache_dir = Path(cache_dir or Path(base) / "articles")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_per_feed = max_articles_per_feed
        self._http = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def close(self):
        await self._http.aclose()

    async def fetch_feeds(self) -> list[Article]:
        """Fetch all configured RSS feeds and return deduplicated articles."""
        tasks = [self._fetch_one(key, info) for key, info in self.sources.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles: list[Article] = []
        seen_urls: set[str] = set()
        for result in results:
            if isinstance(result, Exception):
                print(f"  [feed error] {result}")
                continue
            for article in result:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    articles.append(article)
        return articles

    async def _fetch_one(self, source_key: str, info: dict) -> list[Article]:
        """Fetch a single RSS feed and parse entries."""
        url = info["url"]
        label = info.get("label", source_key)
        source_level = info.get("level", "intermediate")

        resp = await self._http.get(url)
        resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        articles: list[Article] = []

        for entry in feed.entries[: self.max_per_feed]:
            article_url = entry.get("link", "")
            if not article_url:
                continue

            title = entry.get("title", "Untitled")
            summary = entry.get("summary", "") or entry.get("description", "")
            # Strip HTML from summary
            summary = re.sub(r"<[^>]+>", "", summary).strip()

            article = Article(
                title=title,
                url=article_url,
                source=label,
                summary=summary[:500],
                difficulty=source_level,
                fetched_at=datetime.now(timezone.utc).isoformat(),
            )
            articles.append(article)

        return articles

    async def extract_content(self, article: Article) -> Article:
        """Fetch the full article HTML and extract readable text."""
        try:
            resp = await self._http.get(article.url)
            resp.raise_for_status()
            html = resp.text

            doc = ReadabilityDoc(html)
            content_html = doc.summary()
            article.title = doc.title() or article.title

            # Strip HTML tags for plain text
            text = re.sub(r"<[^>]+>", "", content_html)
            text = re.sub(r"\s+", " ", text).strip()
            article.content = content_html
            article.text = text
            article.word_count = len(text.split())

        except Exception as e:
            print(f"  [extract error] {article.url[:60]}: {e}")
            article.text = article.summary
            article.word_count = len(article.summary.split())

        return article

    def get_cached_articles(self, max_age_hours: int = 24) -> list[Article]:
        """Load cached articles from disk if recent enough."""
        articles: list[Article] = []
        cache_file = self.cache_dir / "today.json"
        if not cache_file.exists():
            return articles
        try:
            data = json.loads(cache_file.read_text())
            fetched = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
            age = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
            if age > max_age_hours:
                return articles
            for item in data.get("articles", []):
                articles.append(Article.from_dict(item))
        except Exception:
            pass
        return articles

    def cache_articles(self, articles: list[Article]):
        """Save fetched articles to disk cache."""
        cache_file = self.cache_dir / "today.json"
        data = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "count": len(articles),
            "articles": [a.to_dict() for a in articles],
        }
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ── Difficulty Grading ──────────────────────────────────────────────

# Simple heuristic-based difficulty grader (no API call needed)
# Uses: avg word length, % of words > 6 letters, Flesch-Kincaid approximation

COMMON_CET4_WORDS: set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    # Extended CET-4 common words
    "about", "above", "accept", "across", "act", "active", "activity", "add", "admit",
    "advertise", "afford", "after", "again", "age", "agree", "ahead", "aim", "air",
    "allow", "almost", "alone", "along", "already", "also", "although", "always",
    "among", "amount", "ancient", "anger", "announce", "annual", "another", "answer",
    "anxious", "any", "apart", "appear", "apply", "approach", "area", "argue",
    "arise", "arrange", "arrive", "article", "artist", "ask", "aspect", "assess",
    "assign", "assist", "assume", "atmosphere", "attach", "attack", "attempt",
    "attend", "attitude", "attract", "audience", "author", "available", "average",
    "avoid", "award", "aware", "base", "basic", "basis", "bear", "beat", "beauty",
    "become", "begin", "behave", "behavior", "behind", "believe", "belong",
    "benefit", "beyond", "bill", "birth", "bit", "blame", "blank", "blind",
    "block", "blow", "board", "boil", "bone", "border", "born", "borrow",
    "bottom", "bound", "brain", "branch", "brand", "brave", "bread", "break",
    "breath", "breed", "bridge", "brief", "bright", "broad", "broadcast",
    "broken", "brother", "brown", "brush", "budget", "build", "burn", "burst",
    "business", "button", "cake", "calculate", "call", "calm", "camera",
    "campaign", "cancel", "candidate", "capable", "capacity", "capital",
    "capture", "carbon", "care", "career", "careful", "carry", "case",
    "cash", "cast", "catch", "category", "cause", "celebrate", "central",
    "century", "certain", "chain", "chair", "chairman", "challenge", "champion",
    "chance", "change", "channel", "chapter", "character", "charge", "charity",
    "chart", "check", "chemical", "chief", "child", "choice", "choose",
    "church", "circle", "circumstance", "citizen", "city", "civil", "claim",
    "class", "classic", "classroom", "clean", "clear", "clearly", "climb",
    "clock", "close", "closely", "clothes", "cloud", "club", "coach", "coal",
    "coast", "code", "coffee", "coin", "cold", "collect", "college", "color",
    "combine", "comfort", "command", "comment", "commercial", "commission",
    "commit", "committee", "common", "communicate", "community", "company",
    "compare", "compete", "competition", "competitive", "complain", "complete",
    "complex", "component", "computer", "concentrate", "concept", "concern",
    "conclude", "condition", "conduct", "conference", "confidence", "confirm",
    "conflict", "confuse", "connect", "conscious", "consequence", "conservative",
    "consider", "consist", "constant", "construct", "consumer", "contact",
    "contain", "contemporary", "content", "contest", "context", "contract",
    "contrast", "contribute", "control", "controversy", "convention", "conversation",
    "convince", "cook", "cool", "cooperate", "cope", "copy", "core", "corner",
    "corporate", "correct", "cost", "cotton", "count", "country", "county",
    "couple", "courage", "course", "court", "cousin", "cover", "crack",
    "craft", "crash", "create", "creative", "creature", "credit", "crew",
    "crime", "criminal", "crisis", "criteria", "critical", "criticism",
    "crop", "cross", "crowd", "crucial", "cultural", "culture", "cup",
    "cure", "curious", "current", "curriculum", "custom", "customer",
    "cut", "cycle", "daily", "damage", "dance", "danger", "dangerous",
    "dare", "data", "date", "daughter", "day", "dead", "deal", "dear",
    "death", "debate", "debt", "decade", "decide", "decision", "declare",
    "decline", "deep", "defeat", "defend", "defense", "define", "definite",
    "degree", "delay", "deliberate", "deliver", "demand", "democracy",
    "demonstrate", "deny", "department", "departure", "depend", "dependent",
    "deposit", "describe", "desert", "deserve", "design", "designer",
    "desire", "desk", "desperate", "despite", "destination", "destroy",
    "detail", "detect", "determine", "develop", "device", "devote",
    "dialogue", "diet", "differ", "difference", "different", "difficult",
    "dig", "digital", "dimension", "dinner", "direct", "direction",
    "director", "dirty", "disadvantage", "disagree", "disappear",
    "disappoint", "discipline", "discover", "discuss", "disease",
    "dismiss", "display", "dispute", "distance", "distant", "distinct",
    "distinguish", "distribute", "district", "disturb", "diverse",
    "divide", "division", "doctor", "document", "dollar", "domestic",
    "dominate", "door", "double", "doubt", "down", "downtown", "dozen",
    "draft", "drag", "drama", "dramatic", "draw", "drawing", "dream",
    "dress", "drink", "drive", "driver", "drop", "drug", "dry", "due",
    "during", "dust", "duty", "dynamic", "each", "eager", "ear", "early",
    "earn", "earth", "ease", "east", "eastern", "easy", "eat", "economic",
    "economy", "edge", "edit", "edition", "editor", "educate", "education",
    "effect", "effective", "efficiency", "effort", "egg", "eight", "either",
    "elderly", "elect", "election", "electric", "electronic", "element",
    "eliminate", "else", "elsewhere", "emerge", "emergency", "emission",
    "emotion", "emotional", "emphasis", "empire", "employ", "employee",
    "employer", "employment", "empty", "enable", "encounter", "encourage",
    "end", "enemy", "energy", "enforce", "engage", "engine", "engineer",
    "enjoy", "enormous", "enough", "ensure", "enter", "enterprise",
    "entertain", "enthusiasm", "entire", "entirely", "entrance", "entry",
    "environment", "episode", "equal", "equipment", "era", "error",
    "escape", "especially", "essay", "essential", "establish", "estate",
    "estimate", "evaluate", "evening", "event", "eventually", "ever",
    "every", "evidence", "evil", "evolution", "exact", "exactly",
    "examine", "example", "exceed", "excellent", "except", "exception",
    "exchange", "excite", "exciting", "exclude", "excuse", "execute",
    "executive", "exercise", "exhibit", "exhibition", "exist", "existence",
    "expand", "expansion", "expect", "expense", "expensive", "experience",
    "experiment", "expert", "explain", "explanation", "explore", "explosion",
    "export", "expose", "express", "expression", "extend", "extension",
    "extensive", "extent", "external", "extra", "extraordinary", "extreme",
    "eye", "fabric", "face", "facility", "fact", "factor", "factory",
    "faculty", "fade", "fail", "failure", "fair", "fairly", "faith",
    "fall", "false", "familiar", "family", "famous", "fan", "fantasy",
    "far", "farm", "farmer", "fashion", "fast", "fat", "fatal", "father",
    "fault", "favor", "favorite", "fear", "feature", "federal", "fee",
    "feed", "feedback", "feel", "feeling", "fellow", "female", "fence",
    "festival", "fever", "fiction", "field", "fifteen", "fifty", "fight",
    "figure", "file", "fill", "film", "final", "financial", "find",
    "fine", "finger", "finish", "fire", "firm", "first", "fish", "fit",
    "five", "fix", "flag", "flame", "flat", "flee", "flesh", "flexible",
    "flight", "float", "flood", "floor", "flow", "flower", "fly", "focus",
    "folk", "follow", "following", "food", "foot", "football", "force",
    "foreign", "forest", "forever", "forget", "form", "formal", "former",
    "formula", "forth", "fortune", "forward", "found", "foundation",
    "founder", "four", "frame", "framework", "free", "freedom", "freeze",
    "frequency", "frequent", "fresh", "friend", "friendly", "friendship",
    "frighten", "frog", "from", "front", "frozen", "fruit", "frustrate",
    "fuel", "full", "fun", "function", "fund", "fundamental", "funeral",
    "funny", "fur", "furniture", "further", "gain", "game", "gap",
    "garage", "garden", "gas", "gate", "gather", "gene", "general",
    "generate", "generation", "generous", "genetic", "gentle", "gentleman",
    "genuine", "gesture", "giant", "gift", "gigantic", "girl", "give",
    "glad", "glance", "glass", "global", "glove", "goal", "god",
    "gold", "golden", "golf", "good", "govern", "government", "governor",
    "grab", "grace", "grade", "gradually", "graduate", "grain", "grand",
    "grandfather", "grandmother", "grant", "grass", "grateful", "grave",
    "great", "green", "grey", "ground", "group", "grow", "growth",
    "guarantee", "guard", "guess", "guest", "guide", "guilty", "gun",
    "guy", "habit", "hair", "half", "hall", "hand", "handle", "handsome",
    "hang", "happen", "happy", "harbor", "hard", "hardly", "harm",
    "harmony", "harsh", "harvest", "hat", "hate", "head", "headline",
    "headquarters", "health", "healthy", "hear", "heart", "heat",
    "heaven", "heavy", "height", "hello", "help", "helpful", "hence",
    "heritage", "hero", "hide", "high", "highlight", "highly", "highway",
    "hill", "hint", "hip", "hire", "historic", "historical", "history",
    "hit", "hobby", "hold", "hole", "holiday", "holy", "home", "honest",
    "honor", "hook", "hope", "horizon", "horror", "horse", "hospital",
    "host", "hot", "hotel", "hour", "house", "household", "housing",
    "huge", "human", "humor", "hundred", "hunger", "hungry", "hunt",
    "hurry", "hurt", "husband", "hypothesis", "ice", "idea", "ideal",
    "identify", "identity", "ignore", "ill", "illegal", "illness",
    "illustrate", "image", "imagination", "imagine", "immediate",
    "immigrant", "impact", "implement", "implication", "imply", "import",
    "importance", "important", "impose", "impossible", "impress",
    "impression", "impressive", "improve", "improvement", "incentive",
    "incident", "include", "including", "income", "incorporate",
    "increase", "increasingly", "incredible", "independent", "index",
    "indicate", "individual", "industrial", "industry", "inevitable",
    "infant", "infection", "inflation", "influence", "inform", "informal",
    "information", "ingredient", "initial", "initiative", "injury",
    "inner", "innocent", "innovation", "input", "inquiry", "insect",
    "inside", "insight", "insist", "inspect", "inspire", "install",
    "instance", "instant", "instead", "instinct", "institute",
    "institution", "instruction", "instrument", "insurance", "intellectual",
    "intelligence", "intend", "intense", "intention", "interaction",
    "interest", "interesting", "internal", "international", "internet",
    "interpret", "intervention", "interview", "intimate", "introduce",
    "introduction", "invade", "invasion", "invest", "investigate",
    "investment", "investor", "invitation", "invite", "involve",
    "iron", "island", "isolate", "issue", "item", "itself",
    "jacket", "jail", "jam", "jet", "jewel", "job", "join", "joint",
    "joke", "journal", "journalist", "journey", "joy", "judge",
    "judgment", "juice", "jump", "junior", "jury", "just", "justice",
    "justify", "keen", "keep", "key", "kick", "kid", "kill", "kind",
    "king", "kiss", "kitchen", "knee", "knife", "knock", "know",
    "knowledge", "label", "labor", "lack", "lady", "lake", "land",
    "landscape", "language", "large", "largely", "last", "late",
    "later", "Latin", "latter", "laugh", "launch", "law", "lawyer",
    "lay", "layer", "lead", "leader", "leadership", "leaf", "league",
    "lean", "learn", "learning", "least", "leather", "leave", "lecture",
    "left", "leg", "legacy", "legal", "legend", "legislation",
    "legitimate", "leisure", "lend", "length", "lesson", "let",
    "letter", "level", "liberal", "library", "license", "lie",
    "life", "lifestyle", "lifetime", "lift", "light", "like",
    "likely", "limit", "limited", "line", "link", "lip", "list",
    "listen", "literally", "literary", "literature", "little",
    "live", "living", "load", "loan", "local", "locate", "location",
    "lock", "log", "logic", "lonely", "long", "look", "lord",
    "lose", "loss", "lost", "lot", "loud", "love", "lovely",
    "lower", "loyal", "luck", "lucky", "lunch", "lung",
    "machine", "mad", "magazine", "magic", "mail", "main",
    "mainly", "maintain", "maintenance", "major", "majority",
    "make", "maker", "male", "manage", "management", "manager",
    "manner", "manufacture", "manufacturer", "many", "map",
    "march", "margin", "mark", "market", "marketing", "marriage",
    "married", "marry", "mask", "mass", "massive", "master",
    "match", "mate", "material", "math", "matter", "mature",
    "maximum", "may", "maybe", "mayor", "meal", "mean", "meaning",
    "means", "meantime", "measure", "measurement", "meat", "mechanism",
    "media", "medical", "medicine", "medium", "meet", "meeting",
    "member", "membership", "memory", "mental", "mention", "menu",
    "merchant", "mercy", "merely", "merge", "merit", "message",
    "metal", "method", "meter", "middle", "midnight", "might",
    "migrant", "mild", "mile", "military", "milk", "mill",
    "million", "mind", "mine", "mineral", "minimum", "minister",
    "minor", "minority", "minute", "miracle", "mirror", "miss",
    "missile", "mission", "mistake", "mix", "mixture", "mobile",
    "mode", "model", "moderate", "modern", "modest", "mom",
    "moment", "money", "monitor", "month", "mood", "moon",
    "moral", "more", "moreover", "morning", "mortgage", "most",
    "mostly", "mother", "motion", "motivate", "motor", "mount",
    "mountain", "mouse", "mouth", "move", "movement", "movie",
    "much", "multiple", "murder", "muscle", "museum", "music",
    "musical", "musician", "mutual", "mystery", "myth",
    "naked", "name", "narrative", "narrow", "nasty", "nation",
    "national", "native", "natural", "naturally", "nature",
    "near", "nearby", "nearly", "neat", "necessarily", "necessary",
    "neck", "need", "negative", "negotiate", "negotiation",
    "neighbor", "neighborhood", "neither", "nerve", "nervous",
    "net", "network", "never", "nevertheless", "new", "news",
    "newspaper", "next", "nice", "night", "nine", "noble",
    "nobody", "nod", "noise", "nominate", "none", "nonetheless",
    "nor", "norm", "normal", "normally", "north", "northern",
    "nose", "not", "note", "notebook", "nothing", "notice",
    "notion", "novel", "now", "nowhere", "nuclear", "number",
    "numerous", "nurse", "nut", "object", "objection", "objective",
    "obligation", "observe", "observer", "obstacle", "obtain",
    "obvious", "occasion", "occupy", "occur", "ocean", "odd",
    "off", "offense", "offensive", "offer", "office", "officer",
    "official", "often", "oil", "okay", "old", "olympic",
    "on", "once", "one", "ongoing", "onion", "online",
    "only", "onto", "open", "opening", "operate", "operation",
    "operator", "opinion", "opponent", "opportunity", "oppose",
    "opposite", "opposition", "option", "or", "orange", "orbit",
    "order", "ordinary", "organ", "organic", "organization",
    "organize", "orientation", "origin", "original", "other",
    "otherwise", "ought", "outcome", "outline", "output",
    "outside", "outstanding", "overcome", "overlook", "overseas",
    "owe", "own", "owner", "pace", "pack", "package", "page",
    "pain", "painful", "paint", "painter", "painting", "pair",
    "palace", "pale", "pan", "panel", "panic", "paper",
    "parent", "park", "parliament", "part", "participant",
    "participate", "particular", "particularly", "partly",
    "partner", "partnership", "party", "pass", "passage",
    "passenger", "passion", "passive", "passport", "past",
    "path", "patience", "patient", "pattern", "pause",
    "pay", "payment", "peace", "peak", "peer", "penalty",
    "people", "per", "perceive", "percent", "percentage",
    "perception", "perfect", "perfectly", "perform",
    "performance", "perhaps", "period", "permanent",
    "permission", "permit", "person", "personal", "personality",
    "personally", "personnel", "perspective", "persuade",
    "pet", "phase", "phenomenon", "philosophy", "phone",
    "photo", "photograph", "photographer", "phrase",
    "physical", "physically", "physician", "piano",
    "pick", "picture", "piece", "pile", "pilot",
    "pine", "pink", "pipe", "pitch", "place", "plain",
    "plan", "plane", "planet", "planning", "plant",
    "plastic", "plate", "platform", "play", "player",
    "plea", "plead", "pleasant", "please", "pleased",
    "pleasure", "plenty", "plot", "plus", "pocket",
    "poem", "poet", "poetry", "point", "poison",
    "police", "policy", "political", "politician",
    "politics", "poll", "pollution", "pool", "poor",
    "pop", "popular", "population", "port", "portrait",
    "pose", "position", "positive", "possess",
    "possession", "possibility", "possible", "possibly",
    "post", "potato", "potential", "pound", "pour",
    "poverty", "powder", "power", "powerful",
    "practical", "practice", "praise", "pray",
    "prayer", "precise", "predict", "prefer",
    "preference", "pregnancy", "pregnant", "premise",
    "premium", "preparation", "prepare", "prescribe",
    "prescription", "presence", "present",
    "presentation", "preserve", "president",
    "press", "pressure", "pretend", "pretty",
    "prevail", "prevent", "previous", "previously",
    "price", "pride", "priest", "primary", "prime",
    "principal", "principle", "print", "prior",
    "priority", "prison", "prisoner", "privacy",
    "private", "privilege", "prize", "probably",
    "problem", "procedure", "proceed", "process",
    "produce", "producer", "product", "production",
    "profession", "professional", "professor",
    "profile", "profit", "program", "progress",
    "project", "prominent", "promise", "promote",
    "prompt", "proof", "proper", "properly",
    "property", "proportion", "proposal",
    "propose", "proposed", "prospect", "protect",
    "protection", "protein", "protest",
    "proud", "prove", "provide", "provider",
    "province", "provision", "psychological",
    "psychologist", "psychology", "pub",
    "public", "publication", "publicity",
    "publish", "publisher", "pull", "punishment",
    "purchase", "pure", "purpose", "pursue",
    "push", "put", "qualify", "quality",
    "quantity", "quarter", "queen", "question",
    "queue", "quick", "quickly", "quiet",
    "quietly", "quit", "quite", "quote",
    "race", "racial", "radical", "radio",
    "rail", "rain", "raise", "random",
    "range", "rank", "rapid", "rapidly",
    "rare", "rarely", "rate", "rather",
    "rating", "ratio", "raw", "reach",
    "react", "reaction", "read", "reader",
    "reading", "ready", "real", "realistic",
    "reality", "realize", "really", "reason",
    "reasonable", "recall", "receipt", "receive",
    "recent", "recently", "recipe", "recognition",
    "recognize", "recommend", "record",
    "recording", "recover", "recovery",
    "recruit", "red", "reduce", "reduction",
    "refer", "reference", "reflect",
    "reform", "refugee", "refuse",
    "regard", "regarding", "regardless",
    "regime", "region", "regional", "register",
    "regret", "regular", "regulate", "regulation",
    "reinforce", "reject", "relate", "relation",
    "relationship", "relative", "relatively",
    "relax", "release", "relevant", "relief",
    "religion", "religious", "reluctant", "rely",
    "remain", "remark", "remarkable", "remedy",
    "remember", "remind", "remote", "remove",
    "render", "rent", "repair", "repeat",
    "replace", "report", "reporter", "represent",
    "representative", "republic", "reputation",
    "request", "require", "requirement",
    "rescue", "research", "researcher",
    "reservation", "reserve", "residence",
    "resident", "resign", "resist",
    "resistance", "resolution", "resolve",
    "resort", "resource", "respect",
    "respond", "response", "responsibility",
    "responsible", "rest", "restaurant",
    "restore", "restrict", "restriction",
    "result", "resume", "retail",
    "retain", "retire", "retirement",
    "return", "reveal", "revenue",
    "reverse", "review", "revolution",
    "reward", "rhetoric", "rhythm",
    "rich", "rid", "ride", "rifle",
    "right", "ring", "riot", "rise",
    "risk", "ritual", "rival", "river",
    "road", "rock", "role", "roll",
    "romantic", "roof", "room", "root",
    "rope", "rough", "roughly", "round",
    "route", "routine", "row", "royal",
    "rub", "rule", "run", "runner",
    "running", "rural", "rush",
    "sacred", "sacrifice", "sad", "safe",
    "safety", "sail", "sake", "salad",
    "salary", "sale", "salt", "same",
    "sample", "sand", "satellite",
    "satisfaction", "satisfy", "save",
    "saving", "scale", "scan", "scandal",
    "scare", "scattered", "scene", "schedule",
    "scheme", "scholar", "scholarship",
    "school", "science", "scientific",
    "scientist", "scope", "score", "scream",
    "screen", "script", "search", "season",
    "seat", "second", "secret", "secretary",
    "section", "sector", "secure", "security",
    "see", "seed", "seek", "segment",
    "select", "selection", "self", "sell",
    "senate", "senator", "send", "senior",
    "sense", "sensitive", "sentence",
    "separate", "sequence", "series",
    "serious", "seriously", "servant",
    "serve", "service", "session", "set",
    "setting", "settle", "settlement",
    "seven", "several", "severe",
    "sexual", "shade", "shadow", "shake",
    "shall", "shape", "share", "sharp",
    "shed", "sheet", "shelf", "shell",
    "shelter", "shift", "shine", "ship",
    "shirt", "shock", "shoe", "shoot",
    "shop", "shopping", "shore", "short",
    "shortly", "shot", "should",
    "shoulder", "shout", "show", "shower",
    "shrug", "shut", "sick", "side",
    "sight", "sign", "signal", "signature",
    "significance", "significant", "silence",
    "silent", "silk", "silly", "silver",
    "similar", "similarly", "simple",
    "simply", "simulate", "single",
    "sink", "sir", "sister", "site",
    "situation", "six", "size", "skill",
    "skin", "sky", "slave", "sleep",
    "slice", "slide", "slight", "slightly",
    "slip", "slow", "slowly", "small",
    "smart", "smell", "smile", "smoke",
    "smooth", "snap", "snow", "so-called",
    "soak", "social", "society", "soft",
    "software", "soil", "solar", "soldier",
    "sole", "solid", "solution", "solve",
    "some", "somebody", "somehow",
    "someone", "something", "sometimes",
    "somewhat", "son", "song", "soon",
    "sophisticated", "sorry", "sort",
    "soul", "sound", "soup", "source",
    "south", "southern", "space",
    "span", "spare", "speak", "speaker",
    "special", "specialist", "species",
    "specific", "specifically", "speech",
    "speed", "spell", "spend", "spending",
    "sphere", "spirit", "spiritual",
    "split", "spokesman", "sponsor",
    "sport", "spot", "spread",
    "spring", "square", "stable", "staff",
    "stage", "stair", "stake", "stand",
    "standard", "standing", "star",
    "stare", "start", "state", "statement",
    "station", "statistics", "status",
    "stay", "steady", "steal", "steel",
    "step", "stick", "still", "stimulus",
    "stock", "stomach", "stone", "stop",
    "storage", "store", "storm", "story",
    "straight", "strange", "stranger",
    "strategic", "strategy", "stream",
    "street", "strength", "strengthen",
    "stress", "stretch", "strict",
    "strike", "string", "strip", "stroke",
    "strong", "strongly", "structure",
    "struggle", "student", "studio",
    "study", "stuff", "stupid", "style",
    "subject", "submit", "subsequent",
    "substance", "substantial", "substitute",
    "succeed", "success", "successful",
    "successfully", "such", "suck",
    "sudden", "suddenly", "sue", "suffer",
    "sufficient", "sugar", "suggest",
    "suggestion", "suicide", "suit",
    "suitable", "sum", "summer", "summit",
    "sun", "super", "supermarket", "supply",
    "support", "supporter", "suppose",
    "sure", "surely", "surface",
    "surgery", "surprise", "surprised",
    "surprising", "surround", "survey",
    "survival", "survive", "suspect",
    "suspend", "sustain", "swallow",
    "swap", "swear", "sweep", "sweet",
    "swim", "swing", "switch", "symbol",
    "sympathy", "symptom", "system",
    "table", "tackle", "tail", "take",
    "tale", "talent", "talk", "tank",
    "tap", "tape", "target", "task",
    "taste", "tax", "tea", "teach",
    "teacher", "teaching", "team",
    "tear", "technical", "technique",
    "technology", "teen", "teenager",
    "telephone", "television", "tell",
    "temperature", "temple", "temporary",
    "ten", "tend", "tendency",
    "tension", "tent", "term",
    "terminal", "terrible", "territory",
    "terror", "terrorist", "test",
    "testify", "testing", "text",
    "textbook", "than", "thank",
    "thanks", "that", "theater",
    "theft", "theme", "then",
    "theory", "therapist", "therapy",
    "there", "therefore", "thick",
    "thin", "thing", "think",
    "thinking", "third", "thirty",
    "this", "thorough", "though",
    "thought", "thousand", "threat",
    "threaten", "three", "throat",
    "through", "throughout", "throw",
    "thumb", "thus", "ticket",
    "tide", "tie", "tight", "till",
    "time", "tiny", "tip", "tire",
    "tired", "tissue", "title",
    "today", "together", "toilet",
    "tomorrow", "tone", "tongue",
    "tonight", "tool", "tooth",
    "top", "topic", "total",
    "totally", "touch", "tough",
    "tour", "tourist", "toward",
    "towards", "tower", "town",
    "track", "trade", "tradition",
    "traditional", "traffic", "tragedy",
    "trail", "train", "training",
    "transfer", "transform", "transition",
    "translate", "transmission", "transport",
    "transportation", "trap", "trash",
    "travel", "treasure", "treat",
    "treatment", "treaty", "tree",
    "tremendous", "trend", "trial",
    "tribe", "trick", "trip",
    "troop", "tropical", "trouble",
    "truck", "true", "truly",
    "trust", "truth", "try",
    "tube", "tune", "turn",
    "twice", "twin", "twist",
    "two", "type", "typical",
    "typically", "ugly", "ultimate",
    "ultimately", "umbrella", "unable",
    "uncle", "uncover", "under",
    "undergo", "undergraduate", "undermine",
    "understand", "understanding", "undertake",
    "unemployment", "unexpected", "unfold",
    "unfortunately", "uniform", "union",
    "unique", "unit", "unite",
    "united", "unity", "universal",
    "universe", "university", "unknown",
    "unless", "unlike", "unlikely",
    "until", "unusual", "update",
    "upon", "upper", "upset",
    "urban", "urge", "urgent",
    "use", "used", "useful",
    "user", "usual", "usually",
    "utility", "vacation", "valid",
    "valley", "valuable", "value",
    "variable", "variation", "variety",
    "various", "vary", "vast",
    "vehicle", "venture", "version",
    "versus", "vertical", "vessel",
    "veteran", "via", "victim",
    "victory", "video", "view",
    "viewer", "village", "violate",
    "violation", "violence", "violent",
    "virtually", "virtue", "virus",
    "visible", "vision", "visit",
    "visitor", "visual", "vital",
    "voice", "volume", "volunteer",
    "vote", "voter", "vs",
    "vulnerable", "wage", "wait",
    "wake", "walk", "wall",
    "wander", "want", "war",
    "warm", "warn", "warning",
    "wash", "waste", "watch",
    "water", "wave", "way",
    "weak", "weakness", "wealth",
    "wealthy", "weapon", "wear",
    "weather", "web", "website",
    "wedding", "week", "weekend",
    "weekly", "weight", "welcome",
    "welfare", "well", "west",
    "western", "wet", "what",
    "whatever", "wheel", "when",
    "whenever", "where", "whereas",
    "whether", "which", "while",
    "whisper", "white", "who",
    "whole", "whom", "whose",
    "why", "wide", "widely",
    "widespread", "wife", "wild",
    "will", "willing", "win",
    "wind", "window", "wine",
    "wing", "winner", "winter",
    "wipe", "wire", "wisdom",
    "wise", "wish", "with",
    "withdraw", "within", "without",
    "witness", "woman", "wonder",
    "wonderful", "wood", "wooden",
    "word", "work", "worker",
    "workforce", "working", "workplace",
    "workshop", "world", "worldwide",
    "worry", "worth", "would",
    "wound", "wrap", "write",
    "writer", "writing", "written",
    "wrong", "yard", "yeah",
    "year", "yell", "yellow",
    "yes", "yesterday", "yet",
    "yield", "young", "youngster",
    "youth", "zone",
}


def grade_difficulty(text: str) -> str:
    """Grade article difficulty using heuristics: avg word length + avg sentence length.

    Formulates a composite score = avg_word_len * 2 + avg_sent_len * 0.3.
    Thresholds tuned for CET-4 (beginner) to advanced learners.
    """
    if not text or len(text) < 50:
        return "unknown"

    words = re.findall(r"[a-zA-Z]+", text)
    if not words:
        return "unknown"

    # Average word length
    avg_word_len = sum(len(w) for w in words) / len(words)

    # Average sentence length (meaningful sentences only)
    sentences = re.split(r"[.!?]+", text)
    sentences = [
        s.strip() for s in sentences
        if len(s.strip().split()) > 3  # at least 4 words
    ]
    avg_sent_len = (
        sum(len(s.split()) for s in sentences) / len(sentences)
        if sentences else 0
    )

    # Composite score
    score = avg_word_len * 2 + avg_sent_len * 0.3

    if score < 14:
        return "beginner"
    elif score < 20:
        return "intermediate"
    else:
        return "advanced"


def extract_new_words(text: str, existing_vocab: set | None = None) -> list[dict]:
    """Extract potentially unfamiliar words (beyond CET-4) from text."""
    existing = existing_vocab or set()
    words = re.findall(r"[a-zA-Z]+", text.lower())
    unique = set(w for w in words if len(w) > 5)

    candidates = []
    for w in sorted(unique):
        if w not in COMMON_CET4_WORDS and w not in existing:
            candidates.append({
                "word": w,
                "context": _find_sentence(text, w),
            })

    # Return top 10 most useful new words (longer = more likely advanced)
    candidates.sort(key=lambda x: len(x["word"]), reverse=True)
    return candidates[:10]


def _find_sentence(text: str, word: str) -> str:
    """Find the sentence containing a given word."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for s in sentences:
        if word in s.lower():
            return s.strip()[:120]
    return ""
