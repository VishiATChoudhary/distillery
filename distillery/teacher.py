import hashlib
import json
import os
from pathlib import Path

import httpx


class Teacher:
    """Big model behind any OpenAI-compatible chat API (OpenRouter, OpenAI, vLLM, Ollama...).

    Every answer is cached on disk, so re-running an experiment never pays twice.
    """

    def __init__(self, model, system="", base_url=None, api_key=None, cache_dir=".cache/teacher"):
        self.model = model
        self.system = system
        self.base_url = (base_url or os.environ.get("TEACHER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("TEACHER_API_KEY", "")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def ask(self, prompt, temperature=0.0, max_tokens=512):
        key = json.dumps([self.base_url, self.model, self.system, prompt, temperature, max_tokens])
        hit = self.cache / f"{hashlib.sha256(key.encode()).hexdigest()}.txt"
        if hit.exists():
            return hit.read_text()

        messages = [{"role": "system", "content": self.system}] if self.system else []
        messages.append({"role": "user", "content": prompt})
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            timeout=300,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        hit.write_text(text)
        return text


class MockTeacher:
    """Offline stand-in for dry runs: no key, no network, deterministic, learnable."""

    def ask(self, prompt, temperature=0.0, max_tokens=512):
        return f"Short answer: {prompt.strip().rstrip('?').lower()}."
