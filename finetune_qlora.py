# BiharOne — QLoRA instruction fine-tuning for Llama 3 8B
#
# What this does: loads Llama 3 8B in 4-bit, attaches LoRA adapters, trains on
# our multilingual service dialogues (JSONL chat format), exports to GGUF for Ollama.
#
# Hardware: one GPU with >=8GB VRAM (e.g. T4/3060). Runs on Google Colab free tier.
#
# Install (run once, in the terminal, not here):
#   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#   pip install --no-deps trl peft accelerate bitsandbytes datasets
#
# NOTE: Unsloth/TRL APIs change often. If an argument errors, check the current
# Unsloth notebook at https://github.com/unslothai/unsloth — the structure below
# (load 4-bit -> add LoRA -> format -> train -> export) stays the same.

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# Config
BASE_MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"   # pre-quantized 4-bit Llama 3
DATA_FILE = "sample_sft.jsonl"                          # swap for your full dataset
MAX_SEQ_LEN = 2048
OUTPUT_LORA = "biharone_lora"
OUTPUT_GGUF = "biharone_gguf"

# Load model in 4-bit  (this is the "Q" in QLoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BASE_MODEL,
    max_seq_length = MAX_SEQ_LEN,
    dtype = None,
    load_in_4bit = True,
)

# Attach LoRA adapters  (the ~1% of params we actually train)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                      # adapter rank / capacity
    lora_alpha = 16,             # scaling
    lora_dropout = 0,
    bias = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Load and format data
# Each line is: {"messages": [{"role": "...", "content": "..."}, ...]}
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def to_text(batch):
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in batch["messages"]
    ]
    return {"text": texts}

dataset = dataset.map(to_text, batched=True)

# Train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LEN,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,    # effective batch size 8
        warmup_steps = 5,
        num_train_epochs = 2,               # small data -> few epochs, avoid memorizing
        learning_rate = 2e-4,
        bf16 = torch.cuda.is_bf16_supported(),
        fp16 = not torch.cuda.is_bf16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        logging_steps = 1,
        seed = 3407,
        output_dir = "training_logs",
    ),
)

trainer.train()

# Save adapter
model.save_pretrained(OUTPUT_LORA)
tokenizer.save_pretrained(OUTPUT_LORA)

# Quick test  (does it answer in the right language/tone?)
FastLanguageModel.for_inference(model)
msgs = [{"role": "user", "content": "हमरा जाति प्रमाण पत्र बनवावे के बा"}]
inputs = tokenizer.apply_chat_template(msgs, tokenize=True,
                                       add_generation_prompt=True,
                                       return_tensors="pt").to("cuda")
out = model.generate(input_ids=inputs, max_new_tokens=128, use_cache=True)
print(tokenizer.batch_decode(out)[0])

# Export to GGUF for Ollama
model.save_pretrained_gguf(OUTPUT_GGUF, tokenizer, quantization_method="q4_k_m")

# Then in the terminal, create a Modelfile pointing at the .gguf:
#   FROM ./biharone_gguf/unsloth.Q4_K_M.gguf
# and run:
#   ollama create biharone -f Modelfile
#   ollama run biharone