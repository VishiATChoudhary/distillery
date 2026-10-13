"""The ONE file the research agent edits. Each experiment = one version of this file."""
from dataclasses import dataclass


@dataclass
class Recipe:
    # Teacher: big model behind an API. Its answers become the training data.
    teacher_model: str = "qwen/qwen3-235b-a22b-2507"
    teacher_system: str = "Answer correctly and concisely."
    teacher_temperature: float = 0.0
    judge_model: str = "qwen/qwen3-235b-a22b-2507"

    # Student: small open-weights model we fine-tune with LoRA.
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_rank: int = 16

    # Training. epochs=0 gives the untrained baseline score.
    n_train: int = 1000
    epochs: int = 2
    lr: float = 2e-4
    batch_size: int = 8
    max_len: int = 1024
    max_new_tokens: int = 384
