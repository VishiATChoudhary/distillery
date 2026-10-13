"""Runs on the GPU box. Pure function in, plain dicts out, so any backend can ship it."""
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Student:
    """Small open-weights model + LoRA adapter, trained to imitate the teacher."""

    def __init__(self, model_name, lora_rank=16):
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        lora = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, target_modules="all-linear", task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora)

    def _format(self, prompt):
        if self.tok.chat_template:
            msgs = [{"role": "user", "content": prompt}]
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return f"User: {prompt}\nAssistant: "

    def _loss(self, batch, max_len):
        """Cross-entropy on the teacher's answer tokens only (prompt tokens masked with -100)."""
        ids, labels = [], []
        for ex in batch:
            p = self.tok(self._format(ex["prompt"]), add_special_tokens=False).input_ids
            a = self.tok(ex["response"] + self.tok.eos_token, add_special_tokens=False).input_ids
            ids.append((p + a)[:max_len])
            labels.append(([-100] * len(p) + a)[:max_len])
        n = max(map(len, ids))
        pad = self.tok.pad_token_id
        t = lambda rows: torch.tensor(rows, device=self.device)
        return self.model(
            input_ids=t([x + [pad] * (n - len(x)) for x in ids]),
            attention_mask=t([[1] * len(x) + [0] * (n - len(x)) for x in ids]),
            labels=t([y + [-100] * (n - len(y)) for y in labels]),
        ).loss

    def train(self, pairs, epochs, lr, batch_size, max_len):
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.model.train()
        pairs, losses = list(pairs), []
        for _ in range(epochs):
            random.shuffle(pairs)
            for i in range(0, len(pairs), batch_size):
                loss = self._loss(pairs[i : i + batch_size], max_len)
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses.append(loss.item())
        return sum(losses[-10:]) / len(losses[-10:]) if losses else float("nan")

    @torch.no_grad()
    def ask(self, prompt, max_new_tokens=256):
        self.model.eval()
        enc = self.tok(self._format(prompt), return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True).strip()

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


def train_and_generate(pairs, eval_prompts, recipe, out_dir):
    """The whole GPU job: fine-tune on teacher pairs, answer eval prompts, save adapter."""
    student = Student(recipe["student_model"], recipe["lora_rank"])
    loss = student.train(pairs, recipe["epochs"], recipe["lr"], recipe["batch_size"], recipe["max_len"])
    answers = [student.ask(p, recipe["max_new_tokens"]) for p in eval_prompts]
    student.save(out_dir)
    return {"loss": loss, "answers": answers}
