import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--adapter", required=True)   # folder with adapter_model.safetensors + adapter_config.json
p.add_argument("--out", required=True)
args = p.parse_args()

print("Loading base:", args.base)
base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16, device_map="auto")

print("Attaching LoRA from:", args.adapter)
model = PeftModel.from_pretrained(base, args.adapter)

print("Merging and unloading...")
model = model.merge_and_unload()   # bake LoRA into base weights (fp16)

print("Saving to:", args.out)
tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out, safe_serialization=True)
tok.save_pretrained(args.out)
print("Done:", args.out)