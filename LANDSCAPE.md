# Landscape: who else is distilling (as of 14 Oct 2026)

Star counts and last-push dates pulled from the GitHub API on 14 Oct 2026. Company facts from press and
vendor pages; vendor claims marked as such.

## 1. Startups selling "small model from your big-model API"

| Company | What they sell | Status | Overlap with distillery |
|---|---|---|---|
| **Distil Labs** (Berlin) | Task prompt + ~50 examples in, downloadable task-specific SLM out. Synthetic data, filtering, fine-tune, eval, all automated. Has a Claude Code CLI plugin | Pre-seed (NAP, Helloworld, Project A). Hiring in 2026 "to meet demand" | Closest product twin. Closed source, hosted |
| **OpenPipe** | Capture production LLM traffic, fine-tune/distill smaller models, now RL for agents (ART) | Acquired by CoreWeave, Sep 2025. Lives on as CoreWeave/W&B "Model Distillation" | Same thesis, enterprise, closed |
| **Predibase** | Fine-tune and serve open SLMs on proprietary data (LoRA Land) | Acquired by Rubrik, Jul 2025, ~$109M | Platform, closed |
| **Arcee AI** | SLM company. DistillKit open source, Arcee Cloud hosted | DistillKit last push May 2026, 1.1k stars | Logit/hidden-state KD, white-box only |
| **Thinking Machines (Tinker)** | Training-as-API. Ships on-policy distillation recipes; Inkling Small was built by distilling Inkling | Live, 4.2k stars cookbook | Best backend for white-box KD. Could be a distillery `Backend` |
| **Malted AI** (UK) | Distill large-model output into domain SLMs for enterprises | £6M seed, 2024 | Services-heavy |
| **TensorZero** | Open-source LLM gateway + observability; distill from logged production traffic | 11.7k stars, Apache 2.0, VC-backed | Strongest open-source overlap. Data comes from your gateway logs, not synthesized; no research loop |
| **Kiln AI** | Desktop app: synthetic data, fine-tuning, evals, distill reasoning models, zero-code | 5.2k stars, app source-available, lib MIT | Closest to the planned distillery UI |
| **Smallest.ai**, **Sarvam**, etc. | Vertical SLM companies | Funded | Not tooling |

## 2. Hyperscaler features (free with the platform, locked to it)

| Platform | Feature |
|---|---|
| OpenAI | Stored completions -> Model Distillation (Oct 2024). Teacher and student both OpenAI |
| AWS Bedrock | Model Distillation: Claude / Llama / Nova teachers into Llama / Nova students |
| Azure AI Foundry | Stored completions + distillation, preview, Microsoft and select OSS students |
| Google Vertex | Distillation for Gemini / Gemma |
| CoreWeave / W&B | Full management API: datasets, relabeling, fine-tunes, evals (ex-OpenPipe) |

## 3. Open-source toolkits

| Repo | Stars | Last push | What | Gap vs distillery |
|---|---|---|---|---|
| karpathy/autoresearch | 97.9k | Mar 2026 | agent edits train.py, keep/revert loop | pretraining only, no teacher |
| microsoft/RD-Agent | 14.9k | Oct 2026 | R&D agent, FT-Agent (FT-Dojo paper) | general ML, heavy |
| tensorzero/tensorzero | 11.7k | Jun 2026 | gateway + distill from prod logs | needs their gateway, Rust + ClickHouse |
| huggingface/ml-intern | 10.8k | archived Sep 2026 | agent that runs HF training jobs | dead |
| OpenPipe/ART | 10.8k | Oct 2026 | GRPO for agents | RL not distillation |
| Kiln-AI/Kiln | 5.2k | Oct 2026 | app: synth data + fine-tune + evals | no autonomous loop, no on-policy |
| thinking-machines-lab/tinker-cookbook | 4.2k | Oct 2026 | on/off-policy distillation recipes | Tinker only, white-box |
| argilla-io/distilabel | 3.4k | Oct 2026 | synthetic data pipelines | data only |
| e-p-armstrong/augmentoolkit | 1.9k | Jun 2026 | custom LLMs from documents | data-centric |
| meta-llama/synthetic-data-kit | 1.6k | Oct 2025 | QA pair / CoT generation CLI | data only |
| arcee-ai/DistillKit | 1.1k | May 2026 | logit + hidden-state KD | white-box only |
| nolabs-ai/deepfabric | 0.9k | Oct 2026 | topic-graph synth data -> Unsloth/TRL | data only; closest "pipeline" |
| modelscope/easydistill | 0.5k | Sep 2026 | Alibaba PAI: black+white box KD, SFT, RL | config-driven, no loop |
| predibase/llm_distillation_playbook | 0.6k | 2024 | best practices doc | doc |
| Nardien/agent-distillation | 0.3k | 2025 | paper code | research |

## 4. What nobody ships (the gap)

1. **Autoresearch loop around distillation.** Every tool above is a pipeline you configure once. None has
   an agent that edits the recipe, measures, keeps or reverts, and runs overnight.
2. **Black-box on-policy.** SDFT / GAD style training from text-only teachers is in papers (2025-2026)
   and nowhere in product. TensorZero, Kiln, Distil Labs all do plain SFT on teacher text.
3. **Teacher-agnostic + backend-agnostic in one small codebase.** TensorZero needs its gateway; Tinker
   needs Tinker; hyperscalers need their cloud. distillery is ~600 lines, any API, any GPU.
4. **Failure-driven data.** Personalised-distillation rounds (student attempts, teacher fixes) exist in
   Distil Labs' marketing language only.

## 5. Tailwinds and headwinds

- YC's Garry Tan (Sep 2026): wants US labs distilling frontier models, "do nothing" on regulation.
- Reported July 2026 push by Anthropic's CEO against industrial-scale distillation (secondhand; verify).
  Closed-API terms still forbid training competitors. Open-weight teachers (Qwen, DeepSeek, Inkling
  Small, Llama) are the safe default.
- Small-model learnability research (ACL 2025) says mid-size open teachers beat frontier teachers for
  sub-3B students anyway, so the ToS-safe path is also the better path.

## Sources

- https://github.com/tensorzero/tensorzero · https://github.com/Kiln-AI/Kiln · https://github.com/nolabs-ai/deepfabric
- https://github.com/modelscope/easydistill · https://github.com/arcee-ai/DistillKit · https://github.com/karpathy/autoresearch
- https://coreweave.com/news/coreweave-to-acquire-openpipe-leader-in-reinforcement-learning
- https://techcrunch.com/2025/06/25/rubrik-acquires-predibase-to-accelerate-adoption-of-ai-agents/
- https://techcrunch.com/2025/03/19/nvidia-reportedly-acquires-synthetic-data-startup-gretel
- https://tech.eu/2024/05/15/malted-ai-raises/ · https://www.caplight.com/company/distillabs
- https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/concept-model-distillation
- https://openai.com/index/api-model-distillation/ · https://docs.coreweave.com/model-distillation
- https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/
- https://www.tensorzero.com/blog/distillation-programmatic-data-curation-smarter-llms-5-30x-cheaper-inference/
- https://www.implicator.ai/yc-garry-tan-rejects-distillation-crackdown/
