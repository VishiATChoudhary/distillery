"""The ONE file the research agent edits. Each experiment = one version of this file."""
from dataclasses import dataclass


@dataclass
class Recipe:
    # Teacher: "provider:model". ollama:, openrouter:, openai:, anthropic:, gemini:, groq:, litellm:...
    # (see distillery/teacher.py). Its answers become the training data.
    # Small students learn better from mid-size teachers and short answers (RESEARCH.md, tier 1).
    teacher_model: str = "ollama:qwen2.5:7b"
    teacher_system: str = (
        "Rewrite the user's text in caveman style. Keep every fact, number, name, date and action item. "
        "Drop articles (a/an/the), filler words, pleasantries, hedging. Fragments OK. Short words. "
        "Keep code, URLs and technical terms exact. Output only the rewritten text: no heading, no preamble, no quotes."
    )
    teacher_temperature: float = 0.0
    judge_model: str = "ollama:qwen2.5:7b"

    # Student: small open-weights model we fine-tune with LoRA on all linear layers.
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_rank: int = 16

    # Training. LoRA wants ~10x full-FT lr; small data wants many epochs (tier 0).
    n_train: int = 300
    epochs: int = 8
    lr: float = 5e-4
    batch_size: int = 2
    max_len: int = 512
    max_new_tokens: int = 256

    # Loss: "sft" (copy teacher text) or "sdft" (on-policy self-distillation, tier 3, ~2.5x slower).
    loss: str = "sft"
    sdft_mix: float = 0.5
    sample_tokens: int = 128

    # Rounds > 1: after each round the student answers the train prompts; the teacher rewrites the
    # student's failed attempts and those replace the training targets (tier 2, personalised distillation).
    rounds: int = 1
    refine_threshold: float = 0.6
