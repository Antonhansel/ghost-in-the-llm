#!/usr/bin/env python3
"""
Quick local test of the merged model to see what it generates.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model(model_path="./test_model"):
    print(f"Loading model from {model_path}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True
    )
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    # Test cases
    test_prompts = [
        "A: Salut comment ça va ?\nB: Ça va bien et toi ?\nA:",
        "A: Tu fais quoi ce soir ?\nB: Rien de spécial, et toi ?\nA:",
        "A: J'ai faim\nB: Moi aussi, on commande ?\nA:",
    ]
    
    print("\n" + "="*50)
    print("TESTING MODEL OUTPUT")
    print("="*50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {repr(prompt)}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        # Generate with different parameters
        generation_configs = [
            {"name": "Conservative", "max_new_tokens": 50, "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.2},
            {"name": "Creative", "max_new_tokens": 80, "temperature": 0.9, "top_p": 0.95, "repetition_penalty": 1.1},
        ]
        
        for config in generation_configs:
            print(f"\n{config['name']} generation:")
            
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config["max_new_tokens"],
                    do_sample=True,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    repetition_penalty=config["repetition_penalty"],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode only the new tokens
            generated_text = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
            print(f"Output: {repr(generated_text)}")
            
            # Check if it contains unwanted B: responses
            if "B:" in generated_text:
                print("⚠️  WARNING: Model generated B: response (unwanted)")
            else:
                print("✅ Good: No B: responses generated")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./test_model"
    test_model(model_path)
