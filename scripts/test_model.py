# test_on_cloud.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./outputs")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

# Set pad token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Get model device
device = next(model.parameters()).device
print(f"Model is on device: {device}")

# Test prompts
test_prompts = [
    "A: Salut comment ça va ?\nB: Ça va bien et toi ?\nA:",
    "A: Tu fais quoi ce soir ?\nB: Rien de spécial, et toi ?\nA:",
    "A: J'ai faim\nB: Moi aussi, on commande ?\nA:",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*50}")
    print(f"TEST {i}")
    print(f"{'='*50}")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to same device as model
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    with torch.no_grad():
        # Debug: Check EOS token
        print(f"EOS token: {repr(tokenizer.eos_token)} (ID: {tokenizer.eos_token_id})")
        
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=96,
            temperature=0.7,
            repetition_penalty=1.3,  # Increased to reduce repetition
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    # Post-process: Stop at first B: or double newline
    stop_markers = ["\nB:", "\n\n", "B:"]
    for marker in stop_markers:
        if marker in generated:
            generated = generated.split(marker)[0]
            break
    
    generated = generated.strip()
    
    print(f"Input: {prompt}")
    print(f"Output: {repr(generated)}")
    
    # Check for issues
    original_had_b = "B:" in tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    if original_had_b:
        print("⚠️  WARNING: Model generated B: response (truncated in post-processing)")
    else:
        print("✅ Good: No B: responses generated")
        
    if len(generated.strip()) == 0:
        print("⚠️  WARNING: Empty output")
    elif len(generated.strip()) < 10:
        print("⚠️  WARNING: Very short output")
    else:
        print(f"✅ Generated {len(generated.strip())} characters")