import sys, glob, torch, os
from cog import BasePredictor, Input
from transformers import AutoModelForCausalLM, AutoTokenizer

WEIGHTS_DIR = "weights"

class Predictor(BasePredictor):
    def setup(self):
        try:
            print("Booting. Files in weights/:", len(glob.glob(f"{WEIGHTS_DIR}/*")), file=sys.stderr, flush=True)
            
            # Check if weights directory exists and has required files
            if not os.path.exists(WEIGHTS_DIR):
                raise FileNotFoundError(f"Weights directory '{WEIGHTS_DIR}' not found")
            
            required_files = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
            for file in required_files:
                if not os.path.exists(os.path.join(WEIGHTS_DIR, file)):
                    raise FileNotFoundError(f"Required file '{file}' not found in {WEIGHTS_DIR}")
            
            # Check for safetensors files
            safetensors_files = glob.glob(os.path.join(WEIGHTS_DIR, "*.safetensors"))
            print(f"Found {len(safetensors_files)} safetensors files: {[os.path.basename(f) for f in safetensors_files]}", file=sys.stderr, flush=True)
            if len(safetensors_files) == 0:
                raise FileNotFoundError(f"No safetensors files found in {WEIGHTS_DIR}")
            
            print("Loading tokenizer...", file=sys.stderr, flush=True)
            try:
                self.tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR, local_files_only=True, use_fast=True)
                print("Tokenizer loaded successfully", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"Failed to load tokenizer with use_fast=True, trying use_fast=False: {e}", file=sys.stderr, flush=True)
                self.tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR, local_files_only=True, use_fast=False)
                print("Tokenizer loaded successfully with use_fast=False", file=sys.stderr, flush=True)
            
            # Fix pad token issue - set pad token to eos token but ensure we handle attention masks
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
                print(f"Set pad_token to eos_token: {self.tok.pad_token}", file=sys.stderr, flush=True)
            
            print("Loading model...", file=sys.stderr, flush=True)
            print(f"CUDA available: {torch.cuda.is_available()}", file=sys.stderr, flush=True)
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}", file=sys.stderr, flush=True)
                print(f"Current CUDA device: {torch.cuda.current_device()}", file=sys.stderr, flush=True)
            
            # Load model - Replicate provides consistent GPU environment
            self.model = AutoModelForCausalLM.from_pretrained(
                WEIGHTS_DIR,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            
            print("Model loaded successfully", file=sys.stderr, flush=True)
            
            # Get device information
            try:
                self.device = next(self.model.parameters()).device
                print(f"Model device: {self.device}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"Could not determine model device: {e}", file=sys.stderr, flush=True)
                self.device = torch.device("cpu")
                print(f"Using fallback device: {self.device}", file=sys.stderr, flush=True)
            
            # Clear any cached memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared CUDA cache", file=sys.stderr, flush=True)
            
        except Exception as e:
            print(f"Setup failed with error: {e}", file=sys.stderr, flush=True)
            print(f"Error type: {type(e).__name__}", file=sys.stderr, flush=True)
            import traceback
            print(f"Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
            raise

    def predict(
        self,
        prompt: str = Input(description="Preformatted prompt string"),
        max_new_tokens: int = Input(default=96, ge=1, le=1024),
        temperature: float = Input(default=0.7, ge=0.0, le=2.0),
        top_p: float = Input(default=0.9, ge=0.0, le=1.0),
        top_k: int = Input(default=0, ge=0, le=1000),
        repetition_penalty: float = Input(default=1.2, ge=0.8, le=2.0),
    ) -> str:
        if not prompt:
            raise ValueError("prompt is required")
        
        # Tokenize with attention mask
        inputs = self.tok(prompt, return_tensors="pt", padding=False, truncation=False)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
            )
        return self.tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).rstrip()
