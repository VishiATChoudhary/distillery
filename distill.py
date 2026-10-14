"""Run ONE distillation experiment and print its score.

teacher answers prompts -> student trains on GPU -> judge scores student vs teacher -> results.tsv
With rounds > 1: student answers the train prompts, teacher rewrites its failed attempts, retrain.
"""
import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from distillery import BACKENDS, LLMJudge, MockTeacher, OverlapJudge, make_teacher
from recipe import Recipe

REFINE = (
    "A student answered a question. Rewrite the student's attempt into an answer as good as the reference, "
    "keeping as much of the student's wording and structure as you can. Output only the rewritten answer.\n\n"
    "QUESTION:\n{q}\n\nREFERENCE:\n{ref}\n\nSTUDENT ATTEMPT:\n{ans}"
)


def lines(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local", choices=BACKENDS)
    ap.add_argument("--mock", action="store_true", help="offline dry run: fake teacher, tiny student, overlap judge")
    ap.add_argument("--note", default="", help="one-line description for results.tsv")
    args = ap.parse_args()

    r = Recipe()
    train_prompts, eval_prompts = lines("data/train.txt"), lines("data/eval.txt")
    if args.mock:
        r = replace(r, student_model="HuggingFaceTB/SmolLM2-135M-Instruct", n_train=32, epochs=3,
                    max_new_tokens=32, sample_tokens=32)
        eval_prompts = eval_prompts[:8]
        teacher, judge = MockTeacher(), OverlapJudge()
    else:
        teacher = make_teacher(r.teacher_model, system=r.teacher_system)
        judge = LLMJudge(make_teacher(r.judge_model))
    train_prompts = train_prompts[: r.n_train]
    pool = ThreadPoolExecutor(int(os.environ.get("TEACHER_CONCURRENCY", 16)))

    t0 = time.time()
    ask = lambda p: teacher.ask(p, r.teacher_temperature, r.max_new_tokens)
    train_refs = list(pool.map(ask, train_prompts))
    eval_refs = list(pool.map(ask, eval_prompts))
    pairs = [{"prompt": p, "response": a} for p, a in zip(train_prompts, train_refs)]
    print(f"teacher: {len(pairs)} train pairs, {len(eval_refs)} eval refs ({time.time() - t0:.0f}s)")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    backend = BACKENDS[args.backend]()
    for rnd in range(1, r.rounds + 1):
        probe = train_prompts if rnd < r.rounds else []
        job = {"pairs": pairs, "eval_prompts": eval_prompts, "recipe": asdict(r), "probe_prompts": probe}
        out = backend.run(job, f"{run_id}-r{rnd}")
        if not probe:
            break
        scores = list(pool.map(judge.score, probe, train_refs, out["probe_answers"]))
        failed = [i for i, s in enumerate(scores) if s < r.refine_threshold]
        refine = lambda i: teacher.ask(REFINE.format(q=probe[i], ref=train_refs[i], ans=out["probe_answers"][i]),
                                       r.teacher_temperature, r.max_new_tokens)
        for i, fixed in zip(failed, pool.map(refine, failed)):
            pairs[i] = {"prompt": probe[i], "response": fixed}
        print(f"round {rnd}: train score {sum(scores) / len(scores):.3f}, refined {len(failed)}/{len(probe)} targets")

    scores = list(pool.map(judge.score, eval_prompts, eval_refs, out["answers"]))
    score = sum(scores) / len(scores)
    seconds = time.time() - t0

    for q, ref, ans in list(zip(eval_prompts, eval_refs, out["answers"]))[:3]:
        print(f"\nQ: {q[:200]}\nteacher: {ref[:200]}\nstudent: {ans[:200]}")
    print(f"\nscore: {score:.4f}\nloss: {out['loss']:.4f}\nseconds: {seconds:.0f}\nrun_id: {run_id}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    log = Path("results.tsv")
    if not log.exists():
        log.write_text("run_id\tcommit\tscore\tloss\tseconds\tnote\n")
    with log.open("a") as f:
        f.write(f"{run_id}\t{commit or '-'}\t{score:.4f}\t{out['loss']:.4f}\t{seconds:.0f}\t{args.note}\n")


if __name__ == "__main__":
    main()
