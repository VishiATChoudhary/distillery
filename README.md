# distillery

Distill a big API model (teacher) into a small open model (student) on a rented GPU,
with an agent tuning the recipe in a loop, like karpathy/autoresearch.

```
data/train.txt ──> Teacher.ask() ──> (prompt, answer) pairs
                                          │
                       Backend.run()  ────┘   local GPU/MPS, or Modal cloud GPU
                           │
                     Student.train()   LoRA SFT on teacher answers
                     Student.ask()     answers data/eval.txt
                           │
data/eval.txt ──> Judge.score(student vs teacher) ──> score ──> results.tsv
```

| Class | File | Job |
|---|---|---|
| `Teacher` / `MockTeacher` | `distillery/teacher.py` | Calls any OpenAI-compatible API, caches answers on disk |
| `Student` | `distillery/student.py` | HF model + LoRA, trains on teacher answers, generates |
| `LocalBackend` / `ModalBackend` | `distillery/backends.py` | Runs the GPU job here or on Modal |
| `LLMJudge` / `OverlapJudge` | `distillery/judge.py` | Scores student answers against teacher answers |
| `Recipe` | `recipe.py` | Every knob. The only file the agent edits |

## Run

```bash
uv sync
uv run distill.py --mock --backend local          # offline dry run, no keys, tiny model

# Local teacher (Ollama) + local student (Apple GPU / CUDA). No keys, no ToS worries.
export TEACHER_BASE_URL=http://localhost:11434/v1 TEACHER_API_KEY=ollama TEACHER_CONCURRENCY=4
uv run synth.py --task "verbose work emails and docs" --prefix "Rewrite in caveman style: " --n 300
uv run distill.py --backend local --note baseline

# Cloud teacher + cloud GPU
export TEACHER_API_KEY=... TEACHER_BASE_URL=https://openrouter.ai/api/v1
uv run modal setup                                 # once
uv run distill.py --backend modal --note baseline
```

`RESEARCH.md` ranks the techniques behind the knobs in `recipe.py` (lr rule, epochs on small data,
persona prompt synthesis, failure-refinement rounds, SDFT on-policy loss).

Autonomous loop: open Claude Code here and say "read program.md and start".

GPU card: `DISTILLERY_GPU=H100 uv run distill.py ...` (default A10G).
Trained adapters: `runs/<run_id>` locally, or `modal volume get distillery-runs <run_id>`.

## Teacher terms

Closed APIs (Anthropic, OpenAI, Google) forbid training competing models on their outputs.
Default teacher is open-weights Qwen via OpenRouter for that reason.
