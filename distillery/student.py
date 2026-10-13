"""Runs on the GPU box. Pure function in, plain dicts out, so any backend can ship it."""
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Student:
    """Small open-weights model + LoRA adapter, trained to imitate the teacher.

    Two losses:
      sft:  cross-entropy on the teacher's answer tokens (classic sequence-level distillation).
      sdft: on-policy self-distillation (Shenfeld et al. 2026). The student samples its own answer;
            the same model, shown the teacher's answer in context, is the target distribution for
            every sampled token. Needs no teacher logits, only text. Costs ~2.5x sft.
    """

    def __init__(self, model_name, lora_rank=16):
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        lora = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, target_modules="all-linear", task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora)

    def _format(self, prompt, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        if self.tok.chat_template:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return (f"System: {system}\n" if system else "") + f"User: {prompt}\nAssistant: "

    def _ids(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def _pad(self, rows, value):
        n = max(map(len, rows))
        return torch.tensor([r + [value] * (n - len(r)) for r in rows], device=self.device)

    def _forward(self, prompts, responses):
        """Logits at every response position for each (prompt_ids, response_ids) pair."""
        ids = [p + r for p, r in zip(prompts, responses)]
        mask = [[1] * len(x) for x in ids]
        out = self.model(input_ids=self._pad(ids, self.tok.pad_token_id), attention_mask=self._pad(mask, 0)).logits
        return [out[i, len(p) - 1 : len(p) - 1 + len(r)] for i, (p, r) in enumerate(zip(prompts, responses))]

    def _sft_loss(self, batch, max_len):
        prompts = [self._ids(self._format(ex["prompt"])) for ex in batch]
        responses = [self._ids(ex["response"] + self.tok.eos_token) for ex in batch]
        responses = [r[: max(1, max_len - len(p))] for p, r in zip(prompts, responses)]
        losses = [
            F.cross_entropy(logits, torch.tensor(r, device=self.device))
            for logits, r in zip(self._forward(prompts, responses), responses)
        ]
        return torch.stack(losses).mean()

    def _sdft_loss(self, batch, max_len, sample_tokens):
        plain = [self._ids(self._format(ex["prompt"])) for ex in batch]
        demo = [
            self._ids(self._format(ex["prompt"], system=f"Answer exactly like this reference answer:\n{ex['response']}"))
            for ex in batch
        ]
        with torch.no_grad():
            self.model.eval()
            enc = self.tok([self._format(ex["prompt"]) for ex in batch], return_tensors="pt", padding=True,
                           padding_side="left", add_special_tokens=False).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=sample_tokens, do_sample=True, temperature=1.0,
                                      top_p=0.95, pad_token_id=self.tok.pad_token_id)
            self.model.train()
            samples = []
            for row in gen[:, enc.input_ids.shape[1] :].tolist():
                row = row[: row.index(self.tok.eos_token_id) + 1] if self.tok.eos_token_id in row else row
                samples.append(row or [self.tok.eos_token_id])
            samples = [s[: max(1, max_len - len(d))] for s, d in zip(samples, demo)]
            target = [F.log_softmax(l, -1) for l in self._forward(demo, samples)]
        student = [F.log_softmax(l, -1) for l in self._forward(plain, samples)]
        # Forward KL(teacher-in-context || student) per token, averaged over the sampled answer.
        return torch.stack([(t.exp() * (t - s)).sum(-1).mean() for t, s in zip(target, student)]).mean()

    def train(self, pairs, epochs, lr, batch_size, max_len, loss="sft", sdft_mix=0.5, sample_tokens=128):
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.model.train()
        pairs, losses = list(pairs), []
        for epoch in range(epochs):
            random.shuffle(pairs)
            t0, n0 = time.time(), len(losses)
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                l = self._sft_loss(batch, max_len)
                if loss == "sdft":
                    l = (1 - sdft_mix) * l + sdft_mix * self._sdft_loss(batch, max_len, sample_tokens)
                l.backward()
                opt.step()
                opt.zero_grad()
                losses.append(l.item())
            ep = losses[n0:]
            print(f"epoch {epoch + 1}/{epochs}: loss {sum(ep) / len(ep):.4f} ({time.time() - t0:.0f}s)", flush=True)
        return sum(losses[-10:]) / len(losses[-10:]) if losses else float("nan")

    @torch.no_grad()
    def ask(self, prompts, max_new_tokens=256, batch_size=8):
        self.model.eval()
        out = []
        for i in range(0, len(prompts), batch_size):
            enc = self.tok([self._format(p) for p in prompts[i : i + batch_size]], return_tensors="pt", padding=True,
                           padding_side="left", add_special_tokens=False).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            out += [t.strip() for t in self.tok.batch_decode(gen[:, enc.input_ids.shape[1] :], skip_special_tokens=True)]
        return out

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


def train_and_generate(pairs, eval_prompts, recipe, out_dir, probe_prompts=()):
    """The whole GPU job: fine-tune on teacher pairs, answer eval (and probe) prompts, save adapter."""
    r = recipe
    student = Student(r["student_model"], r["lora_rank"])
    loss = student.train(pairs, r["epochs"], r["lr"], r["batch_size"], r["max_len"],
                         loss=r["loss"], sdft_mix=r["sdft_mix"], sample_tokens=r["sample_tokens"])
    answers = student.ask(list(eval_prompts) + list(probe_prompts), r["max_new_tokens"])
    student.save(out_dir)
    return {"loss": loss, "answers": answers[: len(eval_prompts)], "probe_answers": answers[len(eval_prompts) :]}
