"""Omni Client — Qwen-Omni multimodal API client.

Supports both local (Jetson Orin) and cloud (DashScope) inference
for the Qwen-Omni series models. Handles vision (image), audio (voice),
and text inputs with streaming voice/text output.

Architecture:
- Local: Qwen2.5-Omni-7B via FastAPI inference server on Orin
- Cloud: Qwen3-Omni-30B via DashScope/Aliyun API
- Automatic routing and fallback
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


class ModelMode(Enum):
    LOCAL = "local"    # Qwen2.5-Omni-7B on Orin
    CLOUD = "cloud"    # Qwen3-Omni-30B via API
    AUTO = "auto"      # Auto-select based on task complexity


@dataclass
class VisionResult:
    """Structured result from a vision analysis request."""
    text: str
    objects: list[str] = field(default_factory=list)
    activity: str = ""
    scene_description: str = ""
    suggested_response: str = ""
    complexity: str = "simple"  # simple|medium|complex

    @classmethod
    def from_json(cls, data: dict) -> "VisionResult":
        return cls(
            text=data.get("text", ""),
            objects=data.get("objects", []),
            activity=data.get("activity", ""),
            scene_description=data.get("scene_description", ""),
            suggested_response=data.get("suggested_response", ""),
            complexity=data.get("complexity", "simple"),
        )


class OmniClient:
    """Multimodal client for Qwen-Omni models.

    Usage:
        client = OmniClient(mode=ModelMode.AUTO)
        result = await client.analyze_scene(image_b64=frame_b64)
        speech = await client.generate_speech(analysis=result)
    """

    def __init__(
        self,
        mode: ModelMode = ModelMode.AUTO,
        local_base_url: str | None = None,
        cloud_api_key: str | None = None,
        cloud_model: str = "qwen3-omni-30b",
        local_model: str = "qwen2.5-omni-7b",
        timeout: float = 30.0,
    ):
        self.mode = mode

        # Local config
        self.local_base_url = (
            local_base_url
            or os.environ.get("OMNI_LOCAL_URL")
            or "http://localhost:8100"
        )

        # Cloud config
        self.cloud_api_key = (
            cloud_api_key
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("QWEN_API_KEY")
            or ""
        )
        self.cloud_model = cloud_model

        self.local_model = local_model
        self.timeout = timeout

        # Lazy HTTP clients
        self._local_client: Optional[httpx.AsyncClient] = None
        self._cloud_client: Optional[httpx.AsyncClient] = None

    async def close(self):
        for client in [self._local_client, self._cloud_client]:
            if client:
                await client.aclose()

    # ── Public API ───────────────────────────────────────────────

    async def analyze_scene(
        self,
        image_b64: str,
        context: str = "",
        mode: Optional[ModelMode] = None,
    ) -> VisionResult:
        """Analyze a camera frame: what objects/activities are visible?

        Returns structured analysis with objects, activity, and
        a suggested English teaching response.

        Args:
            image_b64: Base64-encoded JPEG image
            context: Optional context (previous activity, known objects)
            mode: Force local or cloud (overrides default)
        """
        prompt = self._build_scene_prompt(context)
        response = await self._call_vision(
            image_b64=image_b64,
            prompt=prompt,
            mode=mode or self.mode,
        )
        return self._parse_scene_response(response)

    async def generate_dialogue(
        self,
        analysis: VisionResult,
        child_spoke: str = "",
        history: list[dict] | None = None,
        child_age: int = 5,
        mode: Optional[ModelMode] = None,
    ) -> str:
        """Generate a child-friendly English dialogue response.

        Args:
            analysis: Scene analysis result
            child_spoke: What the child said (if they spoke)
            history: Previous conversation turns
            child_age: Child's age for language level adaptation
            mode: Force local or cloud
        """
        prompt = self._build_dialogue_prompt(
            analysis=analysis,
            child_spoke=child_spoke,
            history=history or [],
            child_age=child_age,
        )
        return await self._call_text(prompt, mode=mode or self.mode)

    # ── Streaming API (for real-time interruptible playback) ──────

    async def stream_response(
        self,
        text: str,
        image_b64: str = "",
        child_audio_b64: str = "",
        mode: Optional[ModelMode] = None,
    ) -> AsyncIterator[dict]:
        """Stream Emma's response — yields chunks as they arrive.

        Each chunk is a dict:
          {"type": "text", "content": "..."}    ← text transcript chunk
          {"type": "audio", "content": b"..."}  ← raw PCM audio bytes
          {"type": "done"}                       ← response complete

        The caller pushes audio chunks to PlaybackController.enqueue()
        as they arrive, enabling low-latency first-word + interruptible
        playback.

        Usage:
            async for chunk in client.stream_response(text="What do you see?"):
                if chunk["type"] == "audio":
                    playback_controller.enqueue(chunk["content"])
                elif chunk["type"] == "text":
                    transcript += chunk["content"]
        """
        m = mode or self.mode
        if m == ModelMode.AUTO:
            m = ModelMode.LOCAL

        if m == ModelMode.LOCAL:
            async for chunk in self._stream_local(text, image_b64, child_audio_b64):
                yield chunk
        else:
            async for chunk in self._stream_cloud(text, image_b64, child_audio_b64):
                yield chunk

    async def _stream_local(
        self, text: str, image_b64: str, audio_b64: str
    ) -> AsyncIterator[dict]:
        """Stream from local Orin Qwen-Omni server."""
        if self._local_client is None:
            self._local_client = httpx.AsyncClient(
                base_url=self.local_base_url,
                timeout=httpx.Timeout(self.timeout),
            )

        payload: dict = {
            "prompt": text,
            "stream": True,
            "return_audio": True,
        }
        if image_b64:
            payload["image_base64"] = image_b64
        if audio_b64:
            payload["audio_base64"] = audio_b64

        try:
            async with self._local_client.stream(
                "POST", "/api/stream",
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise RuntimeError(f"Stream error {resp.status_code}: {body}")

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):].strip()
                    if data_str == "[DONE]":
                        yield {"type": "done"}
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Text chunk
                    if "text" in chunk:
                        yield {"type": "text", "content": chunk["text"]}

                    # Audio chunk (base64-encoded WAV bytes)
                    if "audio" in chunk:
                        audio_bytes = base64.b64decode(chunk["audio"])
                        yield {"type": "audio", "content": audio_bytes}

        except Exception as e:
            # Fallback to non-streaming + TTS
            print(f"[WARN] Stream failed: {e}. Falling back...")
            text_result = await self._call_local_text(text)
            yield {"type": "text", "content": text_result}
            yield {"type": "done"}

    async def _stream_cloud(
        self, text: str, image_b64: str, audio_b64: str
    ) -> AsyncIterator[dict]:
        """Stream from cloud DashScope Qwen-Omni API."""
        if self._cloud_client is None:
            self._cloud_client = httpx.AsyncClient(
                base_url="https://dashscope-intl.aliyuncs.com",
                headers={"Authorization": f"Bearer {self.cloud_api_key}"},
                timeout=httpx.Timeout(self.timeout),
            )

        content_parts: list[dict] = [{"text": text}]
        if image_b64:
            content_parts.insert(0, {"image": f"data:image/jpeg;base64,{image_b64}"})
        if audio_b64:
            content_parts.insert(0, {"audio": f"data:audio/wav;base64,{audio_b64}"})

        payload = {
            "model": self.cloud_model,
            "input": {
                "messages": [{"role": "user", "content": content_parts}]
            },
            "parameters": {
                "result_format": "json",
                "stream": True,
                "modalities": ["text", "audio"],
            },
        }

        try:
            async with self._cloud_client.stream(
                "POST", "/compatible-mode/v1/chat/completions",
                json=payload,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):].strip()
                    if data_str == "[DONE]":
                        yield {"type": "done"}
                        break

                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choice = event.get("output", {}).get("choices", [{}])[0]
                    delta = choice.get("delta", {})

                    if "content" in delta:
                        yield {"type": "text", "content": delta["content"]}
                    if "audio" in delta:
                        audio_bytes = base64.b64decode(delta["audio"]["data"])
                        yield {"type": "audio", "content": audio_bytes}

        except Exception as e:
            print(f"[WARN] Cloud stream failed: {e}")
            text_result = await self._call_cloud_text(text)
            yield {"type": "text", "content": text_result}
            yield {"type": "done"}


    async def generate_teaching_moment(
        self,
        analysis: VisionResult,
        target_vocabulary: list[str] | None = None,
        mode: Optional[ModelMode] = None,
    ) -> str:
        """Generate a proactive teaching intervention.

        Called when decision engine determines it's a good moment
        to teach something (child not focused, looking around, etc.)
        """
        vocab = ", ".join(target_vocabulary) if target_vocabulary else "age-appropriate"
        prompt = (
            f"The child (age 4-7) is doing this: {analysis.activity or 'playing'}.\n"
            f"Scene: {analysis.scene_description}\n"
            f"Objects visible: {', '.join(analysis.objects[:5])}\n"
            f"Target vocabulary to practice: {vocab}\n\n"
            f"Generate ONE very short, encouraging English sentence "
            f"that invites interaction. Keep it under 10 words.\n"
            f"Use exclamation marks and enthusiasm.\n"
            f"Example: 'Wow! A red car! Vroom vroom!'"
        )
        return await self._call_text(prompt, mode=mode or self.mode)

    async def generate_game_prompt(
        self,
        game_type: str = "i_spy",
        known_objects: list[str] | None = None,
        mode: Optional[ModelMode] = None,
    ) -> str:
        """Generate a simple game prompt for child interaction.

        Supported games: i_spy, simon_says, counting, movement
        """
        objects = known_objects or ["something in the room"]
        prompts = {
            "i_spy": (
                f"I spy with my little eye {', '.join(objects[:3])}.\n"
                f"Generate an 'I spy' game prompt in very simple English. "
                f"One sentence, under 12 words. Example: 'I spy something red! Can you find it?'"
            ),
            "simon_says": (
                f"Generate a Simon Says instruction for a young child. "
                f"One action, under 8 words. Use body parts and simple movements. "
                f"Example: 'Simon says touch your nose!'"
            ),
            "counting": (
                f"The child can see these objects: {', '.join(objects)}.\n"
                f"Generate a counting game prompt. Under 12 words. "
                f"Example: 'Let's count the blocks! One, two, three!'"
            ),
            "movement": (
                f"Generate a fun movement instruction for a young child. "
                f"Under 8 words. Simple action. "
                f"Example: 'Jump up high! Jump, jump, jump!'"
            ),
        }
        prompt = prompts.get(game_type, prompts["i_spy"])
        return await self._call_text(prompt, mode=mode or self.mode)

    async def check_available(self) -> dict:
        """Check which backends are available.

        Returns: {'local': bool, 'cloud': bool}
        """
        result = {"local": False, "cloud": False}

        # Check local
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.local_base_url}/api/health")
                result["local"] = resp.status_code == 200
        except Exception:
            pass

        # Check cloud
        if self.cloud_api_key:
            result["cloud"] = True

        return result

    # ── Internal: Prompt builders ─────────────────────────────────

    def _build_scene_prompt(self, context: str) -> str:
        return (
            "You are Emma, an English tutor for a young child (age 3-8).\n"
            "Look at this image from a camera overlooking a child's play/study area.\n\n"
            "Respond with a JSON object containing:\n"
            "- objects: list of visible objects (toys, books, furniture, etc.)\n"
            "- activity: what the child is doing (playing, drawing, reading, etc.)\n"
            "- scene_description: one simple English sentence describing the scene\n"
            "- suggested_response: one VERY SHORT English sentence (under 10 words)\n"
            "   that Emma could say to the child about what's happening.\n"
            "   Use enthusiasm! Like: 'Wow! A big red car!'\n"
            "- complexity: 'simple', 'medium', or 'complex' based on scene richness\n\n"
            f"Previous context: {context or 'No previous context'}\n\n"
            "IMPORTANT: Keep all English at child level (A1, ages 3-8).\n"
            "Short sentences. Simple words. Enthusiastic tone."
        )

    def _build_dialogue_prompt(
        self,
        analysis: VisionResult,
        child_spoke: str,
        history: list[dict],
        child_age: int,
    ) -> str:
        age_guidance = {
            3: "Very short sentences (3-5 words). One concept at a time. Lots of repetition.",
            5: "Short sentences (5-8 words). Simple questions. Praise often.",
            7: "Medium sentences (6-10 words). Slightly more complex. Encourage full sentences.",
            9: "Normal sentences (8-12 words). Can use because/when. More natural flow.",
        }
        guidance = age_guidance.get(child_age, age_guidance[5])

        history_str = ""
        if history:
            recent = history[-4:]  # Last 4 turns
            history_str = "\n".join(
                f"{'👧' if h['role'] == 'user' else '🤖 Emma'}: {h['content']}"
                for h in recent
            )

        child_line = f"The child just said: '{child_spoke}'" if child_spoke else ""

        return (
            f"You are Emma, a warm, encouraging English tutor for a {child_age}-year-old child.\n"
            f"Guidance: {guidance}\n\n"
            f"The child is currently: {analysis.activity or 'nearby'}\n"
            f"The child can see: {', '.join(analysis.objects[:3]) or 'various things'}\n"
            f"{child_line}\n\n"
            f"Recent conversation:\n{history_str}\n\n"
            f"Generate ONE English response. Follow these rules:\n"
            f"1. Match the child's age ({child_age}).\n"
            f"2. If the child spoke English, acknowledge it positively first.\n"
            f"3. If the child made a grammar error, model the correct version (don't say 'you're wrong').\n"
            f"4. Include ONE simple follow-up question.\n"
            f"5. Sound excited and warm. Use emojis in your thinking but not in speech.\n\n"
            f"Your response (ONE sentence — child hears this spoken):"
        )

    # ── Internal: API calls ───────────────────────────────────────

    async def _call_vision(
        self,
        image_b64: str,
        prompt: str,
        mode: ModelMode,
    ) -> str:
        """Send a vision request to the appropriate backend."""
        if mode == ModelMode.AUTO:
            # Use local for simple, cloud for complex
            # Auto-detection happens at a higher level
            mode = ModelMode.LOCAL  # Default to local

        if mode == ModelMode.LOCAL:
            return await self._call_local_vision(image_b64, prompt)
        else:
            return await self._call_cloud_vision(image_b64, prompt)

    async def _call_text(self, prompt: str, mode: ModelMode) -> str:
        """Send a text-only request."""
        if mode == ModelMode.AUTO:
            mode = ModelMode.LOCAL

        if mode == ModelMode.LOCAL:
            return await self._call_local_text(prompt)
        else:
            return await self._call_cloud_text(prompt)

    async def _call_local_vision(self, image_b64: str, prompt: str) -> str:
        """Send vision request to local Orin inference server."""
        if self._local_client is None:
            self._local_client = httpx.AsyncClient(
                base_url=self.local_base_url,
                timeout=httpx.Timeout(self.timeout),
            )

        try:
            resp = await self._local_client.post(
                "/api/vision",
                json={"image_base64": image_b64, "prompt": prompt},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("text", "")
        except Exception as e:
            # Fallback to cloud if local fails and cloud is available
            if self.cloud_api_key:
                print(f"[WARN] Local vision failed: {e}. Falling back to cloud.")
                return await self._call_cloud_vision(image_b64, prompt)
            raise

    async def _call_cloud_vision(self, image_b64: str, prompt: str) -> str:
        """Send vision request to DashScope Qwen-Omni API."""
        if self._cloud_client is None:
            self._cloud_client = httpx.AsyncClient(
                base_url="https://dashscope-intl.aliyuncs.com",
                headers={"Authorization": f"Bearer {self.cloud_api_key}"},
                timeout=httpx.Timeout(self.timeout),
            )

        # DashScope multimodal format
        payload = {
            "model": self.cloud_model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:image/jpeg;base64,{image_b64}"},
                            {"text": prompt},
                        ],
                    }
                ]
            },
            "parameters": {
                "result_format": "json",
            },
        }

        resp = await self._cloud_client.post(
            "/compatible-mode/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["output"]["choices"][0]["message"]["content"]

    async def _call_local_text(self, prompt: str) -> str:
        """Send text-only request to local Orin server."""
        if self._local_client is None:
            self._local_client = httpx.AsyncClient(
                base_url=self.local_base_url,
                timeout=httpx.Timeout(self.timeout),
            )

        resp = await self._local_client.post(
            "/api/generate",
            json={"prompt": prompt, "max_new_tokens": 100},
        )
        resp.raise_for_status()
        return resp.json().get("text", "")

    async def _call_cloud_text(self, prompt: str) -> str:
        """Send text-only request to cloud API."""
        if self._cloud_client is None:
            self._cloud_client = httpx.AsyncClient(
                base_url="https://dashscope-intl.aliyuncs.com",
                headers={"Authorization": f"Bearer {self.cloud_api_key}"},
                timeout=httpx.Timeout(self.timeout),
            )

        payload = {
            "model": self.cloud_model,
            "input": {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {"result_format": "text"},
        }

        resp = await self._cloud_client.post(
            "/compatible-mode/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["output"]["choices"][0]["message"]["content"]

    # ── Internal: Response parsing ────────────────────────────────

    def _parse_scene_response(self, raw: str) -> VisionResult:
        """Parse a scene analysis response into structured result."""
        try:
            # Try direct JSON
            data = json.loads(raw)
            return VisionResult.from_json(data)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown or mixed text
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return VisionResult.from_json(data)
            except json.JSONDecodeError:
                pass

        # Fallback: treat raw text as scene description
        return VisionResult(
            text=raw,
            scene_description=raw.strip()[:200],
            suggested_response=raw.strip()[:100],
            complexity="simple",
        )
