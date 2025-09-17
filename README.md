# Ghost in the Shell: Your Digital Doppelgänger

A simple, effective approach to creating personalized AI that sounds exactly like you using just your text messages. Inspired by [Edward Donner's experiment](https://edwarddonner.com/2024/01/02/fine-tuning-an-llm-on-240k-text-messages/), but with a focus on simplicity and authenticity.

## 🎯 Project Overview

**Goal**: Fine-tune Mistral-7B on ~70k personal chat messages to replicate your authentic conversation style.

### 📊 Training Results Summary

![All Training Runs](images/all_runs.png)
*Complete training history showing iterative improvements and debugging phases*

![Final Successful Run](images/last_run_only.png)
*Final 8-epoch training run with stable convergence (loss ~1.7)*

| Feature | Description |
|---------|-------------|
| **Input** | Variable context window (6-12 previous messages) |
| **Output** | Next message as you would write it (style transfer, not Q&A) |
| **Model** | Mistral-7B-v0.3 with QLoRA fine-tuning |
| **Training** | 8 epochs, 16h on H100, stable convergence (loss 3.4→1.7) |
| **Cost** | ~$48 final training, ~$200 total with experimentation |
| **Memory** | ~7.8 GiB VRAM, ~2,180 tokens/sec |
| **Deployment** | Replicate or Hugging Face Hub |

## 🚀 Key Features

- **Simple Architecture**: Direct fine-tuning approach - no complex RAG needed
- **Authentic Voice**: Train AI to chat exactly like you using 70k+ messages
- **Multilingual Support**: Natural French/English code-switching
- **Rolling Context**: Maintains conversation flow with last ~20 turns
- **Privacy-First**: Built-in PII masking and data protection
- **Cost-Effective**: QLoRA fine-tuning on cloud GPU

## 🧠 Technical Stack

### Model Architecture
- **Base Model**: `mistralai/Mistral-7B-v0.3`
- **Fine-tuning**: QLoRA (4-bit quantization) with LoRA adapters
- **Quantization**: `nf4` with `bfloat16` compute dtype
- **LoRA Config**: r=16, alpha=32, dropout=0.05
- **Target Modules**: All linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)

### Training Configuration
- **Sequence Length**: 1024 tokens, effective batch size 64
- **Learning Rate**: 5e-5 with cosine scheduling, AdamW optimizer
- **Epochs**: 8, H100 PCIe with mixed precision (bf16)

### Data Format
```jsonl
{"text": "A: So first you can edit messages on telegram So correction messages with an asterisk there you go\nB: Are you giving me lessons? The guy with a hoodie photo? :p Ok ok it's new\nA: You look like a boomer using telegram\nB: I have trouble with my phone\nA: With your GSM 😂\nB: On my cellphone yes xD Coming from a thirty-something it's offensive\nA: Wowowow No insults Thirty-something calm down I'm two years older than you, when is your birthday?</s>"}
```

**Key Principles**:
- **Context**: Variable window (6-12 previous messages)
- **Target**: Next A: message as user would write it
- **Training Type**: Completion training (next-token prediction)
- **Speakers**: A: (you) and B: (everyone else)


## 📁 Project Structure

```
ghost-in-the-shell/
├── 01_message_parsing.ipynb      # Message extraction and preprocessing
├── 02_prepare_messages.ipynb     # Message preparation and JSONL creation
├── data/                         # Raw and processed message data
│   ├── raw/                      # Raw message exports (WhatsApp, Telegram, Messenger)
│   ├── cleaned/                  # Cleaned message data
│   └── processed/                # Training-ready JSONL datasets
├── models/                       # Trained model checkpoints
├── scripts/                      # Utility scripts
│   ├── merge_lora_fp16.py        # LoRA merging script
│   ├── test_model.py             # Model testing script
│   └── sanity_check.py           # Post-training validation
├── replicate_deployment/         # Replicate deployment configuration
├── config/                       # Configuration files
│   ├── training_config.yaml      # QLoRA training configuration
│   └── user_config.json          # User identifier mapping
└── requirements.txt              # Python dependencies
```

## 🎯 Why This Approach Works

### Simple Architecture Philosophy

- **No Complex RAG**: Direct fine-tuning on conversation data is more effective than retrieval-augmented generation
- **Rolling Context**: Maintain conversation flow with the last ~20 message turns
- **JSONL Format**: Simple conversation chunks instead of complex chunking strategies
- **Two Speakers Only**: A: (you) and B: (everyone else) - clean and focused
- **QLoRA Efficiency**: 4-bit quantization enables cost-effective training

### The Complete Pipeline

1. **Parse** → Extract messages from WhatsApp, Telegram, Messenger exports
2. **Clean** → Anonymize while preserving your authentic voice and style
3. **Segment** → Create conversation windows with proper boundaries
4. **Train** → Fine-tune Mistral-7B with QLoRA on your conversations
5. **Deploy** → Serve via Replicate or Hugging Face Hub

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (for training) or cloud GPU access
- 16GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ghost-in-the-shell.git
   cd ghost-in-the-shell
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up data directories**
   ```bash
   mkdir -p data/raw/{whatsapp,messenger,telegram}
   mkdir -p data/processed data/output models config logs
   ```

### Step 1: Configure Your Identity

Edit `config/user_config.json` to identify which senders are YOU:

```json
{
  "user_identifiers": [
    "YourRealName",
    "yourrealname", 
    "YourNickname",
    "yournickname"
  ],
  "user_label": "A:",
  "other_label": "B:",
  "description": "Configuration for identifying the user's messages in chat data."
}
```

### Step 2: Process Your Messages

1. **Export your chat data** from WhatsApp/Telegram/Messenger
2. **Place exports in** `data/raw/whatsapp/` (or appropriate folder)
3. **Run notebook 01**: `01_message_parsing.ipynb` - processes raw data
4. **Run notebook 02**: `02_prepare_messages.ipynb` - creates JSONL training data

### Step 3: Train Your Model

Follow the [Cloud Training Guide](#-cloud-training) to fine-tune Mistral-7B on your data.


## ☁️ Cloud Training

### Why Cloud Training?

- **Cost-effective**: ~$48 final training, ~$200 total with experimentation
- **No Setup**: No local GPU setup required  
- **Scalable**: Access to high-end hardware (H100, A100)
- **Reliable**: Stable training environment

### Training Setup

#### 1. Setup Lambda Labs

1. Create account at [cloud.lambda.ai](https://cloud.lambda.ai/instances)
2. Add billing and SSH key
3. Launch instance: **H100 PCIe** or **A100 40GB**, Lambda Stack 22.04

#### 2. Prepare Environment

```bash
# SSH into your instance
ssh -i ~/.ssh/your_lambda_key ubuntu@<INSTANCE_IP>

# Setup environment
sudo apt update && sudo apt install -y git wget tmux python3.10-venv git-lfs jq
git lfs install
python3 -m venv axo && source axo/bin/activate

# Install axolotl and dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/OpenAccess-AI-Collective/axolotl.git
cd axolotl && pip install -e .
pip install bitsandbytes accelerate transformers datasets peft sentencepiece wandb "huggingface_hub>=0.24" huggingface_hub[cli]
```

#### 3. Upload Your Data

```bash
# From your local machine
scp -i ~/.ssh/your_lambda_key train.jsonl val.jsonl ubuntu@<INSTANCE_IP>:/home/ubuntu/axolotl/
scp -i ~/.ssh/your_lambda_key config/training_config.yaml ubuntu@<INSTANCE_IP>:/home/ubuntu/axolotl/
```

#### 4. Start Training

```bash
# Use tmux to keep training running
tmux new -s train

# Login to services
huggingface-cli login
wandb login

# Start training
accelerate launch -m axolotl.cli.train training_config.yaml
```

> **Tip**: Use `Ctrl+B, then D` to detach from tmux. Rejoin with `tmux attach`.

### Post-Training Steps

### 5. Test Your Model

Before deploying, it's crucial to test your trained model to ensure it's generating authentic responses.

#### Upload Test Script

```bash
# From your local machine, upload the test script
scp -i ~/.ssh/your_lambda_key scripts/test_model.py ubuntu@<INSTANCE_IP>:/home/ubuntu/axolotl/
```

#### Run Model Tests

```bash
# On your instance, activate the environment
source ~/axo/bin/activate

# Navigate to axolotl directory
cd ~/axolotl

# Run the test script
python test_model.py
```

#### Expected Test Results

With good training (loss < 1.0), you should see:

```bash
==================================================
TEST 1
==================================================
Model is on device: cuda:0
Input: A: Do you want to see Dune this weekend?
       B: Yes sounds good I love you too
       A:
Output: "haha I just got the EDF bill we went a bit crazy this winter I think 😂"
✅ Good: No B: responses generated
✅ Generated 68 characters

==================================================
TEST 2  
==================================================
Input: A: Can I finish or do you want some?
       A: I'm hungry answer me hahaha
       B: I'll be home in 10 mins I'm taking the 14 in 10 mins
       A:
Output: "ok :) what do you want to eat? steak & fries?"
✅ Good: No B: responses generated
✅ Generated 45 characters
```

#### Quality Indicators

**Good Signs** ✅:
- Responses sound natural and authentic
- No "B:" responses generated
- Appropriate length (20-80 characters typically)
- Contextually relevant to the conversation
- Maintains your personal style/language

**Warning Signs** ⚠️:
- Very short responses (< 10 characters)
- Generates "B:" responses (conversation continuation issue)
- Repetitive or nonsensical text
- Responses don't match conversation context

#### Troubleshooting

If you encounter issues:

```bash
# Check if model files exist
ls -la merged-mistral7b/

# Verify environment is activated
echo $VIRTUAL_ENV  # Should show: /home/ubuntu/axo

# Install missing dependencies if needed
pip install transformers torch peft bitsandbytes sentencepiece

# Test with verbose output
python test_model.py --verbose
```



#### 5. Merge LoRA Adapters

```bash
# Upload merge script
scp -i ~/.ssh/your_lambda_key scripts/merge_lora_fp16.py ubuntu@<INSTANCE_IP>:/home/ubuntu/axolotl/

# Merge LoRA to FP16
python merge_lora_fp16.py \
  --base mistralai/Mistral-7B-v0.3 \
  --adapter ./outputs/ \
  --out ./merged-mistral7b
```

#### 6. Upload to Hugging Face Hub

```bash
export HF_MODEL_ID=YourUsername/ghost-in-the-shell-mistral7b

huggingface-cli login
huggingface-cli repo create "$HF_MODEL_ID" --private --type model

cd merged-mistral7b
git lfs install --system
git init
git lfs track "*.safetensors" "*.model" "*.json"
git add .gitattributes

git config --global user.email "your.email@example.com"
git config --global user.name "YourUsername"

git add .
git commit -m "Merged FP16 weights"
git branch -m main
git remote add origin https://huggingface.co/$HF_MODEL_ID
git push -u origin main
```

#### 7. Backup and Clean Up

Before shutting down your instance, decide what to save:

**Option A: Minimal (Recommended)**
```bash
# Verify merged model is uploaded to Hugging Face Hub
huggingface-cli repo info YourUsername/ghost-in-the-shell-mistral7b

# If successful, shutdown immediately to save costs
```

**Option B: Complete Backup**
```bash
# 1. Backup training checkpoint (optional insurance)
tar -czf checkpoint-backup.tar.gz outputs/

# 2. Download checkpoint to local machine
scp -i ~/.ssh/your_lambda_key ubuntu@<INSTANCE_IP>:/home/ubuntu/axolotl/checkpoint-backup.tar.gz ./

# 3. Verify merged model on HF Hub, then shutdown
```

**Why Keep Checkpoint?**
- Continue training if needed (more epochs)
- Experiment with different merge settings  
- Backup in case merged model has issues
- Research and analysis

**Cost Consideration**: Each hour costs $3 on H100. If your model is working well (good test results), Option A is recommended.

**Important**: Always shut down your instance in Lambda Labs UI to avoid large bills!

#### 8. Deploy with Replicate

```bash
cd replicate_deployment
chmod +x scripts/download_weights.py
HUGGING_FACE_HUB_TOKEN=*** python ./scripts/download_weights.py

cog build
cog run python ./preflight.py  # Should return all green

# Deploy
cog login
cog push r8.im/your-user/ghost-in-the-llm
```

## 🤖 Deployment Options

- **Replicate API**: Fast cold starts (~few minutes), scalable inference (~3s), simple HTTP API
- **Hugging Face Hub**: Private model hosting, easy transformers integration, version control

## 📊 Results & Performance

### Training Metrics
- **Loss**: 3.4 → 1.7 (stable convergence over 8 epochs)
- **Efficiency**: 7.8 GiB VRAM, 2,180 tokens/sec throughput
- **Quality**: Smooth convergence, no overfitting or gradient explosions

### Quality Achievements
- **Authentic Voice**: Successfully replicates personal conversation style
- **Multilingual**: Natural French/English code-switching  
- **Context Awareness**: Maintains conversation flow with rolling context
- **Style Transfer**: Mimics user's writing patterns, expressions, and casual tone
- **Technical Success**: Stable QLoRA training, clean LoRA→FP16 merge, multiple deployment options

## 🔧 Data Processing Details

### Supported Platforms & Pipeline
- **Platforms**: WhatsApp (TXT), Telegram (JSON), Messenger (JSON)
- **Pipeline**: Extract → Clean → Language detect → Filter → Normalize speakers (A:/B:) → Merge consecutive → Segment conversations → Generate rolling windows → Format JSONL

### Language Filtering
The notebooks use `langdetect` library for language detection. By default, they filter for French conversations, but you can easily modify this for other languages:

```python
# In 01_message_parsing.ipynb - modify language detection
def is_french_conversation(messages, sample_size=20):
    # Change 'fr' to your target language code (e.g., 'en', 'es', 'de')
    french_count = languages.count('fr')  # Change 'fr' to your language
    french_ratio = french_count / len(languages)
    is_french = french_ratio >= 0.6  # Adjust threshold as needed
```

**Why Language Filtering?**
- **Consistent training data** - Single language allows deeper pattern learning
- **Authentic voice replication** - Focus on your primary language
- **Reduced confusion** - Avoid mixed language patterns during training

### Key Features
- **Two Speakers Only**: A: (you) and B: (everyone else)
- **Authentic Preservation**: Keep emojis, slang, natural style
- **Privacy Protection**: Built-in PII masking
- **Conversation Boundaries**: Proper segmentation and context windows

## ⚠️ Known Limitations

### Conversation Step Generation Issue

**The Problem**: The model sometimes generates multiple conversation steps instead of stopping after generating the user's response (A:). Despite training with `</s>` end-of-sequence tokens, the model may generate:

```
A: ok :) what do you want to eat? steak & fries?
B: Yes very good
A: I'm in the 14 But there's a technical incident I'm near pont cardinet there
```

**Expected Behavior**: The model should stop after generating the first A: response:
```
A: ok :) what do you want to eat? steak & fries?</s>
```

**Current Workarounds**:
- **Post-processing**: Test scripts include stop markers (`\nB:`, `\n\n`, `B:`) to truncate unwanted continuations
- **Generation parameters**: Higher `repetition_penalty` (1.2-1.3) helps reduce the issue
- **Manual monitoring**: Check test outputs for "⚠️ WARNING: Model generated B: response"

**Why This Happens**: This appears to be a fundamental challenge with completion-based training on conversational data. Even with proper `</s>` token placement during training, the model learns the conversational pattern so strongly that it continues generating dialogue turns.

**Attempted Solutions That Didn't Work**:
- Training with `</s>` tokens after each A: response
- Adjusting training parameters (learning rate, epochs, batch size)
- Different prompt formatting strategies
- Various generation parameters (temperature, top_p, top_k)

This limitation means **post-processing is currently required** for production deployments to ensure clean single-response generation.

## 🤝 Contributing

This project demonstrates that **simple approaches often work better**. If you have ideas for making it even simpler or more effective, contributions are welcome!

**Particularly welcome**: Solutions to the conversation step generation issue above - this is the main unsolved technical challenge.

## 📄 License

MIT License - feel free to build your own digital doppelgänger!

---

**⚠️ Disclaimer**: This project is for educational and personal use. Always respect privacy laws and obtain proper consent before processing personal data.