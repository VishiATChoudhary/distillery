# Research notes: squeezing a black-box teacher

Setting: teacher gives text only (API). Student is small, open weights, LoRA. Judge is an LLM.
Ranked by expected payoff per unit of effort in this exact setting. Each tier is a plausible
`recipe.py` or `distillery/` change for the autoresearch loop.

## Tier 0: free wins, do first

| Idea | Evidence | What to change |
|---|---|---|
| LoRA lr about 10x the full-FT lr, invariant to rank, apply to all linear layers | Schulman et al., "LoRA Without Regret" (2025). Tinker docs say 20-100x for small models | `lr` sweep 1e-4 .. 1e-3. `target_modules="all-linear"` already set |
| Vanilla LoRA with tuned lr is within 1-2% of DoRA/rsLoRA/LoRA+ | "Learning Rate Matters" (arXiv 2602.04998) | Tune lr, do not add adapter variants |
| LoRA dislikes big batches, penalty not fixed by rank | LoRA Without Regret | `batch_size` 4..16, not 64 |
| Many epochs on few samples beats one epoch on many; stop when train token accuracy saturates | "Data Repetition Beats Data Scaling" (arXiv 2602.11149): 128 epochs x 400 samples beat 1 x 51k by 12-26 pts | Log token accuracy in `Student.train`, let `epochs` go to 10-30 for n_train < 1000 |
| Small nonzero prompt-loss weight (0.01-0.1) helps short answers; irrelevant for long ones | Huerta-Enochian & Ko (arXiv 2401.13586) | Optional `prompt_loss_weight` in `_loss` |
| Judge: score both orders / reference-guided, tell judge to ignore length | "Judging the Judges" (IJCNLP 2025); Zheng et al. 2023 length bias | `LLMJudge` is already reference-guided. Add "ignore length" line |

## Tier 1: data is the lever

| Idea | Evidence | What to change |
|---|---|---|
| Synthesize prompts with personas or Magpie-style sampling for diversity at scale | Persona Hub (arXiv 2406.20094): 7B hits 65% MATH. Magpie (arXiv 2406.08464) beats Evol-Instruct/UltraChat | New `PromptSynth` step: teacher writes N prompts per persona; dedupe |
| Filter for difficulty, diversity, quality; 1k curated > 50k raw | s1 (arXiv 2501.19393), LIMO (arXiv 2502.03387), DEITA/SuperFiltering | Judge scores each train pair, keep top-k; embed-dedupe |
| Small students (<= 3B) learn better from shorter answers and mid-size teachers. Mix 20% strong-teacher data with 80% simpler data | "Small Models Struggle to Learn from Strong Reasoners" (ACL 2025), alpha = 0.2 | `teacher_system`: "concise, direct". Or second cheaper teacher, mix 80/20 |
| Capacity gap: teacher too strong vs student saturates | Distillation Scaling Laws (arXiv 2502.08606) | Try a 7B-32B teacher before a 235B one for a 0.5B student |
| Teacher explanations personalized to what the student gets wrong | "Can LMs Teach Weaker Agents?" (arXiv 2306.09299) | Feed the student's wrong answer into the teacher prompt |

## Tier 2: close the loop on student mistakes (cheap on-policy)

| Idea | Evidence | What to change |
|---|---|---|
| Student attempts first; teacher refines only the failures; train on refinements | Personalised Distillation (EMNLP 2023): beat standard distillation with 1/3 the data, $4-6 of API | Round 2 of loop: `Student.ask` on train prompts, judge, send failures + attempt to teacher, retrain |
| Teacher writes new hard prompts aimed at the student's weak spots | Lion (EMNLP 2023): 70k samples, matched ChatGPT on open-ended gen | Teacher sees (prompt, student answer, score < 0.5), emits 3 harder siblings |
| Sample teacher K times, judge picks best (or majority), train on winner | RFT / best-of-N targets; RARE used K=8 | `teacher_temperature` 0.7, `k_samples` 4, judge-select |

## Tier 3: true on-policy, still black-box

| Idea | Evidence | Cost |
|---|---|---|
| **SDFT**: teacher = the student itself with the teacher's answer in context; student samples, minimize reverse KL to the context-conditioned copy. Needs only student logits | Shenfeld et al., "Self-Distillation Enables Continual Learning" (ICML 2026): beats SFT, far less forgetting | about 2.5x SFT compute. Clean fit: we have teacher text and student logits |
| **GAD**: discriminator (init from student + scalar head) scores teacher vs student text with Bradley-Terry loss; student trained by GRPO on discriminator reward; 1 epoch SFT warmup, N=8 samples/prompt, beta=0.001 | "Black-Box On-Policy Distillation" (arXiv 2511.10643): +1.5 to +2 GPT-4o points over SeqKD, 14B student matched GPT-5-Chat | 16 H100 x 30h at 14B. At 0.5B, hours on one A10G |
| Equal-size discriminator beats bigger one; skip warmup costs 1-3 pts | GAD ablations | Keep D = student size |

## Tier 4: needs an open-weight teacher served with logits (vLLM)

GKD (ICLR 2024), MiniLLM (reverse KL), DistiLLM-2 (skew KL, contrastive), Tinker on-policy recipe
(per-token reverse KL, 76.7% AIME24 with rank-128 LoRA). Best quality per sample, but the teacher
must be yours. Switch `Teacher` to a vLLM endpoint with `logprobs` and add a KD loss in `Student`.

## Order of operations for the agent

1. Tier 0 sweeps (lr, epochs with token-accuracy stop, batch).
2. Tier 1: grow n_train with persona prompts, filter with judge, shorter teacher answers.
3. Tier 2: failure-driven round 2 (biggest gain per API dollar).
4. Tier 3: SDFT loss, then GAD if GPU budget allows.

## Sources

- https://thinkingmachines.ai/blog/lora/
- https://arxiv.org/abs/2602.04998
- https://arxiv.org/abs/2602.11149
- https://arxiv.org/abs/2401.13586
- https://arxiv.org/abs/2406.07791
- https://arxiv.org/abs/2406.20094
- https://arxiv.org/abs/2406.08464
- https://arxiv.org/abs/2501.19393
- https://arxiv.org/abs/2502.03387
- https://arxiv.org/abs/2502.12143
- https://arxiv.org/abs/2502.08606
- https://arxiv.org/abs/2306.09299
- https://arxiv.org/abs/2310.18628
- https://arxiv.org/abs/2305.12870
- https://arxiv.org/abs/2601.19897
- https://arxiv.org/abs/2511.10643
- https://arxiv.org/abs/2604.00626
- https://arxiv.org/abs/2306.13649
- https://arxiv.org/abs/2306.08543
- https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/
