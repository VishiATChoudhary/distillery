# Distillery: autonomous distillation research

You are a research agent. Goal: make the student imitate the teacher as well as possible.
Metric: `score` printed by `distill.py` (0..1, higher is better). Judged on `data/eval.txt`.

## Rules

- Edit ONLY `recipe.py`. Do not touch `data/eval.txt`, `distillery/judge.py`, or `distill.py` (that would be cheating the metric).
- You may add prompts to `data/train.txt` (never copy eval prompts into it).
- Teacher answers are cached; changing `teacher_system`, `teacher_temperature`, `teacher_model` or `max_new_tokens` costs new API calls. Keep `n_train` under 5000 unless told otherwise.
- Bigger student = better score but slower and pricier. Prefer wins that come from data and hyperparameters.

## Setup (once)

1. `git checkout -b distill/<tag>` (e.g. `distill/oct12`).
2. Baseline: set `epochs = 0`, commit, run. This is the untrained student's score.
3. Restore `epochs`, commit, run. This is the first real distillation score.

## Loop (never stop, never ask the human)

1. Pick one idea (see below). Edit `recipe.py`.
2. `git commit -am "<idea>"`
3. `uv run distill.py --backend modal --note "<idea>" > run.log 2>&1`
4. `grep -E "^(score|loss|seconds):" run.log`. If no score, `tail -50 run.log`, fix or abandon.
5. Score improved over the best so far: keep the commit. Otherwise: `git reset --hard HEAD~1`.
6. `results.tsv` (untracked) keeps the full history. Read it before choosing the next idea.

## Ideas

- lr (1e-4 .. 5e-4), epochs (1..5), lora_rank (8..128), batch_size
- n_train: more teacher data usually helps the most
- teacher_system: ask for answers the student can actually copy (shorter, structured, step by step)
- max_len / max_new_tokens: truncation hurts long answers
- student_model: try a bigger base only after data and hyperparameters plateau
