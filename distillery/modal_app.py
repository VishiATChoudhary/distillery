import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "peft", "accelerate")
    .add_local_python_source("distillery")
)
app = modal.App("distillery", image=image)
runs = modal.Volume.from_name("distillery-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("distillery-hf-cache", create_if_missing=True)


@app.function(
    gpu=os.environ.get("DISTILLERY_GPU", "A10G"),
    timeout=3 * 3600,
    volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache},
)
def train_remote(job, run_id):
    from distillery.student import train_and_generate

    out = train_and_generate(**job, out_dir=f"/runs/{run_id}")
    runs.commit()
    return out
