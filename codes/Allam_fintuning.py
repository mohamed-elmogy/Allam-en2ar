import os
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    LlamaTokenizer,
)
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login

# -----------------------------
# Config
# -----------------------------
login(token="Add your Hugging face token here")

model_name   = "ALLaM-AI/ALLaM-7B-Instruct-preview"
tokenizer_name = "ALLaM-AI/ALLaM-7B-Instruct-preview"
Adapter_path = "./allam-en2ar-lora/lora_adapter"
data_path    = "D:/Allam-en2ar-main/Data/train_data_preprocessed.csv"
data_format  = "csv"
en_col, ar_col = "en_clean", "ar_clean"
output_dir   = "./allam-en2ar-lora-v4"
eval_on = True
epochs = 20
batch_size = 8             # safer for memory
grad_accum = 16
lr = 5e-5
max_seq_len = 512
use_4bit = True
fp16 = True

# -----------------------------
# Prompt Template
# -----------------------------
PROMPT_TEMPLATE = (
    "Translate English to Arabic:\n"
    "<EN> {source}\n"
    "<AR> {target} <EOS>"
)
# -----------------------------
# Dataset loading / processing
# -----------------------------

def load_translation_dataset(path: str, fmt: str, en_col: str, ar_col: str):
    if fmt == "csv":
        ds = load_dataset("csv", data_files=path)
    elif fmt in ("json", "jsonl"):
        ds = load_dataset("json", data_files=path)
    elif fmt == "parquet":
        ds = load_dataset("parquet", data_files=path)
    else:
        raise ValueError("Unsupported data_format")

    cols = set(ds["train"].column_names)
    if en_col not in cols or ar_col not in cols:
        raise ValueError(f"Columns not found. Available: {sorted(cols)}")

    def build_examples(example):
        src = str(example[en_col]).strip()
        tgt = str(example[ar_col]).strip()
        prompt = PROMPT_TEMPLATE.format(source=src, target=tgt)
        return {"text": prompt}

    ds = ds.map(build_examples, remove_columns=ds["train"].column_names)
    if "validation" not in ds and eval_on:
        split = ds["train"].train_test_split(test_size=0.02, seed=42)
        ds = {"train": split["train"], "validation": split["test"]}
    return ds


def build_examples(example):
    src = str(example[en_col]).strip()
    tgt = str(example[ar_col]).strip()
    prompt = PROMPT_TEMPLATE.format(source=src, target=tgt)
    return {"text": prompt}

# -----------------------------
# Main
# -----------------------------

def main():
   
    # Tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    quant_config = None
    device_map = "auto"
    torch_dtype = torch.float16
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    # Base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quant_config,
    )
    special_tokens = {"additional_special_tokens": ["<EN>", "<AR>", "<EOS>"]}
    tokenizer.add_special_tokens(special_tokens)

    # Resize model embeddings to handle new tokens
    model.resize_token_embeddings(len(tokenizer))
    # LoRA config
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    )

    # Dataset
    ds = load_translation_dataset(data_path, data_format, en_col, ar_col)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=lr,
    weight_decay=0,
    warmup_ratio=0,
    logging_strategy="steps",
    logging_steps=50,                 # adjust as needed
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    bf16=(not fp16 and torch_dtype == torch.bfloat16),
    fp16=fp16,
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
    report_to=["tensorboard"],
    )


    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=(ds.get("validation") if eval_on else None),
        data_collator=collator,
    )
    class PrintMetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                print(f"Step {state.global_step} - Loss: {logs.get('loss')}, Eval Loss: {logs.get('eval_loss')}")
                
    trainer.add_callback(PrintMetricsCallback)
    trainer.train()

    trainer.model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    tokenizer.save_pretrained(output_dir)

    print("Training finished. LoRA adapter saved to:", os.path.join(output_dir, "lora_adapter"))


if __name__ == "__main__":
    main()
