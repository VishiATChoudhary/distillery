"""Teachers: anything that turns a prompt into text. Pick one with a "provider:model" string.

    ollama:qwen2.5:7b              local, OpenAI-compatible
    openrouter:qwen/qwen3-235b     any OpenAI-compatible host (openai, groq, together, deepseek, mistral,
    openai:gpt-4o                  gemini, vllm, ollama, or a raw URL via TEACHER_BASE_URL)
    anthropic:claude-opus-5-5      official anthropic SDK          (uv sync --extra anthropic)
    litellm:bedrock/anthropic...   100+ providers through LiteLLM  (uv sync --extra litellm)

Every answer is cached on disk by content hash, so re-running an experiment never pays twice.
"""
import hashlib
import json
import os
from pathlib import Path

import httpx

# OpenAI-compatible hosts: name -> (base_url, api key env var). Any of them also works as a judge.
OPENAI_COMPATIBLE = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "vllm": ("http://localhost:8000/v1", None),
    "url": (None, None),  # url:<model> with TEACHER_BASE_URL + TEACHER_API_KEY
}


class Teacher:
    """Base: disk cache + the ask() interface. Subclasses implement _complete()."""

    def __init__(self, model, system="", cache_dir=".cache/teacher"):
        self.model = model
        self.system = system
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def _complete(self, messages, temperature, max_tokens):
        raise NotImplementedError

    def ask(self, prompt, temperature=0.0, max_tokens=512):
        key = json.dumps([type(self).__name__, self.model, self.system, prompt, temperature, max_tokens])
        hit = self.cache / f"{hashlib.sha256(key.encode()).hexdigest()}.txt"
        if hit.exists():
            return hit.read_text()
        messages = [{"role": "system", "content": self.system}] if self.system else []
        messages.append({"role": "user", "content": prompt})
        text = self._complete(messages, temperature, max_tokens)
        hit.write_text(text)
        return text


class OpenAITeacher(Teacher):
    """Any /chat/completions endpoint: OpenAI, OpenRouter, Groq, Together, Gemini, Ollama, vLLM..."""

    def __init__(self, model, system="", base_url=None, api_key=None, **kw):
        super().__init__(model, system, **kw)
        self.base_url = (base_url or os.environ.get("TEACHER_BASE_URL", OPENAI_COMPATIBLE["openrouter"][0])).rstrip("/")
        self.api_key = api_key or os.environ.get("TEACHER_API_KEY", "none")

    def _complete(self, messages, temperature, max_tokens):
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class AnthropicTeacher(Teacher):
    """Claude via the official SDK. Credentials from ANTHROPIC_API_KEY or `ant auth login`.
    Current Claude models reject sampling params and think adaptively by default, so only the
    system prompt, messages and max_tokens are sent."""

    def __init__(self, model, system="", **kw):
        super().__init__(model, system, **kw)
        import anthropic

        self.client = anthropic.Anthropic()

    def _complete(self, messages, temperature, max_tokens):
        extra = {"system": self.system} if self.system else {}
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[m for m in messages if m["role"] != "system"],
            **extra,
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in response.content if block.type == "text")


class LiteLLMTeacher(Teacher):
    """Catch-all: Bedrock, Vertex, Azure, Cohere, Mistral, Replicate... model strings per LiteLLM docs."""

    def __init__(self, model, system="", **kw):
        super().__init__(model, system, **kw)
        import litellm

        self.litellm = litellm

    def _complete(self, messages, temperature, max_tokens):
        out = self.litellm.completion(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return out.choices[0].message.content or ""


class MockTeacher:
    """Offline stand-in for dry runs: no key, no network, deterministic, learnable."""

    def ask(self, prompt, temperature=0.0, max_tokens=512):
        return f"Short answer: {prompt.strip().rstrip('?').lower()}."


def make_teacher(spec, system="", **kw):
    """'provider:model' -> Teacher. No prefix means an OpenAI-compatible host from TEACHER_BASE_URL."""
    provider, _, model = spec.partition(":")
    if not model or provider not in {"anthropic", "litellm", *OPENAI_COMPATIBLE}:
        return OpenAITeacher(spec, system, **kw)
    if provider == "anthropic":
        return AnthropicTeacher(model, system, **kw)
    if provider == "litellm":
        return LiteLLMTeacher(model, system, **kw)
    base_url, key_var = OPENAI_COMPATIBLE[provider]
    api_key = os.environ.get(key_var) if key_var else None
    return OpenAITeacher(model, system, base_url=base_url, api_key=api_key, **kw)
