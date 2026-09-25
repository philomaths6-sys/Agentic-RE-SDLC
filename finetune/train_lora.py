# finetune/train_lora.py
"""Local 4-bit QLoRA Fine-Tuning Pipeline for Qwen-2.5-3B-Instruct.

Trains lightweight LoRA adapters for:
1. Extraction & Classification Agent (schema compliance + MoSCoW prioritization)
2. SDLC Recommendation Agent (methodology ranking + security stage gates)

Optimized for 6GB VRAM (RTX 3050 Laptop GPU):
- 4-bit NormalFloat (NF4) base model
- Gradient checkpointing enabled
- LoRA rank = 16, alpha = 32, dropout = 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj
- Batch size = 1, gradient accumulation = 4
"""

import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BASE_MODEL_ID = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"


def format_instruction(sample, tokenizer):
    """Format instruction, input, and output into chat prompt."""
    messages = [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user", "content": sample["input"]},
        {"role": "assistant", "content": sample["output"]},
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": formatted_text}


def train_adapter(agent_type: str, epochs: int = 2, max_samples: int = 150):
    print(f"\n=======================================================")
    print(f"🚀 Starting 4-bit QLoRA Fine-Tuning for: {agent_type.upper()} AGENT")
    print(f"Base Model: {BASE_MODEL_ID}")
    print(f"=======================================================\n")

    dataset_path = f"data/training/agent_{agent_type}.jsonl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}. Run finetune/prepare_dataset.py first.")

    output_dir = f"finetune/adapters/{agent_type}_adapter"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load and preprocess dataset
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    raw_dataset = raw_dataset.shuffle(seed=42)
    if max_samples and max_samples < len(raw_dataset):
        raw_dataset = raw_dataset.select(range(max_samples))

    formatted_dataset = raw_dataset.map(lambda x: format_instruction(x, tokenizer))

    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            max_length=512,
            truncation=True,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, remove_columns=formatted_dataset.column_names)

    # 3. Load 4-bit Quantized Model
    print("[INFO] Loading base 4-bit model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("\n[INFO] Trainable Parameter Summary:")
    model.print_trainable_parameters()

    # 5. Training Arguments tailored for 6GB VRAM
    training_args = TrainingArguments(
        output_dir=f"finetune/checkpoints/{agent_type}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        num_train_epochs=epochs,
        save_strategy="no",
        report_to="none",
        optim="paged_adamw_8bit",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\n[INFO] Commencing QLoRA training loop...")
    trainer.train()

    # 6. Save Adapter & Tokenizer
    print(f"\n[SUCCESS] Training completed! Saving adapter weights to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[SUCCESS] LoRA adapter for {agent_type} successfully stored.\n")

    # Clean GPU memory
    del model
    del trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import subprocess
    parser = argparse.ArgumentParser(description="Train 4-bit LoRA adapter for financial RE & SDLC agents.")
    parser.add_argument("--agent", choices=["extraction", "sdlc", "all"], default="extraction", help="Which agent adapter to fine-tune")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--samples", type=int, default=60, help="Number of dataset samples to use for fast training")
    args = parser.parse_args()

    if args.agent == "all":
        print("[INFO] Executing Extraction Agent training in dedicated process...")
        subprocess.run([sys.executable, __file__, "--agent", "extraction", "--epochs", str(args.epochs), "--samples", str(args.samples)], check=True)
        print("\n[INFO] Executing SDLC Agent training in dedicated process...")
        subprocess.run([sys.executable, __file__, "--agent", "sdlc", "--epochs", str(args.epochs), "--samples", str(args.samples)], check=True)
    elif args.agent in ["extraction", "sdlc"]:
        train_adapter(args.agent, epochs=args.epochs, max_samples=args.samples)
