#!/usr/bin/env python

import os
from huggingface_hub import snapshot_download

REPO = os.environ.get("REPO_ID", "Antonhansel/ghost-in-the-shell-mistral7b")
REVISION = "master"  # set to "master" explicitly, change to main if you pushed to main
CACHE_DIR = "weights"  # baked into the image

snapshot_download(
    repo_id=REPO,
    revision=REVISION,
    local_dir="weights",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "tokenizer.*",
        "model.safetensors.index.json",
        "*.safetensors",
        "*.json",
    ],
    token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
)
print("Downloaded", REPO, "@", REVISION, "-> weights")
