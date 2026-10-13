"""Run ONE distillation experiment and print its score.

teacher answers prompts -> student trains on GPU -> judge scores student vs teacher -> results.tsv
"""
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from distillery import BACKENDS, LLMJudge, MockTeacher, OverlapJudge, Teacher
from recipe import Recipe


def lines(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="modal", choices=BACKENDS)
    ap.add_argument("--mock", action="store_true", help="offline dry run: fake teacher, tiny student, overlap judge")
    ap.add_argument("--note", default="", help="one-line description for results.tsv")
    args = ap.parse_args()

    r = Recipe()
    train_prompts, eval_prompts = lines("data/train.txt"), lines("data/eval.txt")
    if args.mock:
        r = replace(r, student_model="HuggingFaceTB/SmolLM2-135M-Instruct", n_train=32, epochs=3, max_new_tokens=32)
        eval_prompts = eval_prompts[:8]
        teacher, judge = MockTeacher(), OverlapJudge()
    else:
        teacher = Teacher(r.teacher_model, system=r.teacher_system)
        judge = LLMJudge(Teacher(r.judge_model))
    train_prompts = train_prompts[: r.n_train]

    t0 = time.time()
    ask = lambda p: teacher.ask(p, r.teacher_temperature, r.max_new_tokens)
    with ThreadPoolExecutor(16) as pool:
        train_refs = list(pool.map(ask, train_prompts))
        eval_refs = list(pool.map(ask, eval_prompts))
    pairs = [{"prompt": p, "response": a} for p, a in zip(train_prompts, train_refs)]
    print(f"teacher: {len(pairs)} train pairs, {len(eval_refs)} eval refs ({time.time() - t0:.0f}s)")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    job = {"pairs": pairs, "eval_prompts": eval_prompts, "recipe": asdict(r)}
    out = BACKENDS[args.backend]().run(job, run_id)

    with ThreadPoolExecutor(16) as pool:
        scores = list(pool.map(judge.score, eval_prompts, eval_refs, out["answers"]))
    score = sum(scores) / len(scores)
    seconds = time.time() - t0

    for q, ref, ans in list(zip(eval_prompts, eval_refs, out["answers"]))[:3]:
        print(f"\nQ: {q}\nteacher: {ref[:200]}\nstudent: {ans[:200]}")
    print(f"\nscore: {score:.4f}\nloss: {out['loss']:.4f}\nseconds: {seconds:.0f}\nrun_id: {run_id}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    log = Path("results.tsv")
    if not log.exists():
        log.write_text("run_id\tcommit\tscore\tloss\tseconds\tnote\n")
    with log.open("a") as f:
        f.write(f"{run_id}\t{commit or '-'}\t{score:.4f}\t{out['loss']:.4f}\t{seconds:.0f}\t{args.note}\n")


if __name__ == "__main__":
    main()
