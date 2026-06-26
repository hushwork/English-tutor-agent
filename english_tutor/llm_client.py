"""LLM client for DeepSeek Flash API (OpenAI-compatible)."""

from __future__ import annotations

import os
import json
from typing import AsyncIterator

import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


class LLMClient:
    """Lightweight OpenAI-compatible chat completion client."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. "
                "Create a .env file or export the environment variable."
            )
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")) or DEFAULT_BASE_URL
        self.model = (model or os.environ.get("LLM_MODEL")) or DEFAULT_MODEL
        self._client = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Send a chat completion request and yield tokens as they arrive."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(
                    f"API error {resp.status_code}: {body.decode(errors='replace')}"
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

    async def chat_sync(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Non-streaming variant — returns the full response string."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = await self._client.post("/chat/completions", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(
                f"API error {resp.status_code}: {resp.text}"
            )
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def close(self):
        await self._client.aclose()
