#!/usr/bin/env python3

import os, sys, json
from pathlib import Path

WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", "weights"))

def fail(msg):
    print(f"[PRECHECK] FAIL: {msg}", file=sys.stderr, flush=True);
    sys.exit(1)

print(f"[PRECHECK] Weights dir = {WEIGHTS_DIR.resolve()}")

# 1) must-have files
required = ["config.json", "tokenizer.json", "tokenizer.model"]
missing = [f for f in required if not (WEIGHTS_DIR / f).exists()]
if missing:
    fail(f"Missing files: {missing}")

# 2) config sanity
try:
    cfg = json.loads((WEIGHTS_DIR / "config.json").read_text())
    print(f"[PRECHECK] arch={cfg.get('architectures')}, vocab_size={cfg.get('vocab_size')}")
except Exception as e:
    fail(f"config.json unreadable: {e}")

# 3) tokenizer boot (no model)
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR), local_files_only=True, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        print("[PRECHECK] pad_token_id is None. Will set to eos at runtime.")
    print(f"[PRECHECK] tokenizer ok: {tok.__class__.__name__}")
except Exception as e:
    fail(f"tokenizer load failed: {e}")

# 4) safetensors headers only (no tensor loads)
try:
    from safetensors import safe_open
    shards = sorted(WEIGHTS_DIR.glob("*.safetensors"))
    if not shards:
        fail("no *.safetensors shards found")
    total_keys = 0
    for s in shards:
        with safe_open(str(s), framework="pt", device="cpu") as f:
            keys = list(f.keys())  # header read, no allocation
            total_keys += len(keys)
            print(f"[PRECHECK] shard {s.name}: {len(keys)} tensors (header only)")
    print(f"[PRECHECK] total tensor entries across shards: {total_keys}")
except Exception as e:
    fail(f"safetensors header inspection failed: {e}")

# 5) optional meta build if accelerate is present
try:
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    config = AutoConfig.from_pretrained(str(WEIGHTS_DIR), local_files_only=True)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(config)
    nparams = sum(int(p.numel()) for p in m.parameters())
    print(f"[PRECHECK] meta-build ok: params~{nparams:,}")
except Exception as e:
    print(f"[PRECHECK] meta-build skipped or failed (ok for preflight): {e}")

print("[PRECHECK] OK")
