"""Synthesize diverse prompts with the teacher (persona-driven, RESEARCH.md tier 1).

    uv run synth.py --task "verbose emails, docs and chat messages to compress" --n 300

Writes data/train.txt and data/eval.txt (90/10 split), one prompt per line.
"""
import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from distillery import make_teacher
from recipe import Recipe

PERSONAS = (
    "List {n} very different people who might write or need the following kind of text: {task}. "
    "One per line, each a short description like 'retired nurse writing to her landlord'. No numbering."
)
PROMPTS = (
    "You are: {persona}.\nWrite {k} realistic, self-contained examples of: {task}. "
    "Each example is 60-150 words. Vary topic, tone and format. "
    'Return a JSON list of strings and nothing else.'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="what kind of input text the student will see")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--prefix", default="", help="text prepended to every prompt, e.g. 'Compress into caveman:\\n'")
    ap.add_argument("--personas", type=int, default=40)
    args = ap.parse_args()

    teacher = make_teacher(Recipe().teacher_model, cache_dir=".cache/synth")
    personas = [p.strip("-* ") for p in teacher.ask(PERSONAS.format(n=args.personas, task=args.task),
                                                     temperature=1.0, max_tokens=2000).splitlines() if p.strip()]
    k = -(-args.n // len(personas))

    def gen(persona):
        out = teacher.ask(PROMPTS.format(persona=persona, k=k, task=args.task), temperature=1.0, max_tokens=4000)
        m = re.search(r"\[.*\]", out, re.S)
        try:
            return [s.strip() for s in json.loads(m.group()) if isinstance(s, str) and len(s.split()) > 20]
        except (AttributeError, json.JSONDecodeError):
            return []

    with ThreadPoolExecutor(4) as pool:
        texts = [t for batch in pool.map(gen, personas) for t in batch]
    texts = list(dict.fromkeys(" ".join(t.split()) for t in texts))
    random.seed(0)
    random.shuffle(texts)
    prompts = [args.prefix.replace("\\n", " ") + t for t in texts[: args.n]]
    n_eval = max(8, len(prompts) // 10)
    Path("data/eval.txt").write_text("\n".join(prompts[:n_eval]) + "\n")
    Path("data/train.txt").write_text("\n".join(prompts[n_eval:]) + "\n")
    print(f"{len(personas)} personas, {len(texts)} unique texts, {len(prompts) - n_eval} train / {n_eval} eval")


if __name__ == "__main__":
    main()
