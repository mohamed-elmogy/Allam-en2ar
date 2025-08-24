import argparse
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
)
from peft import LoraConfig
from trl import SFTTrainer

model_name = 'ALLaM-AI/ALLaM-7B-Instruct-preview'
data_path = 'Data/data_preprocessed.csv'
data_format = 'csv'
en_col = 'en_clean'
ar_col = 'ar_clean'
output_dir = './allam-en2ar-lora'
epochs = 3
batch_size = 2
grad_accum  = 16
lr = 2e-4
max_seq_len  = 1024
use_4bit = True
weight_decay = 0
warmup_ratio = 0
logging_steps = 50
fp16 = True
eval_on = 'epochs'
resume_from = 'ALLaM-AI/ALLaM-7B-Instruct-preview'

# -----------------------------
# Prompt Template
# -----------------------------
PROMPT_TEMPLATE = (
    "Translate the following sentence from English to Arabic.\n\n"
    "English: {source}\n"
    "Arabic:"  # The answer (target) will be appended after this tag
)

RESPONSE_TAG = "Arabic:"  # Used by the collator to mask everything before the answer

# -----------------------------
# Dataset loading / processing
# -----------------------------

def load_translation_dataset(path: str, fmt: str, en_col: str, ar_col: str, args):
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
        prompt = PROMPT_TEMPLATE.format(source=src)
        return {"text": f"{prompt} {tgt}"}

    ds = ds.map(build_examples, remove_columns=ds["train"].column_names)
    if "validation" not in ds and eval_on:
        split = ds["train"].train_test_split(test_size=0.02, seed=42)
        ds = {"train": split["train"], "validation": split["test"]}
    return ds


# -----------------------------
# Main
# -----------------------------

def main():
   
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
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

    # LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
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
    weight_decay=weight_decay,
    warmup_ratio=warmup_ratio,
    logging_strategy="steps",
    logging_steps=50,                 # adjust as needed
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    bf16=(not fp16 and torch_dtype == torch.bfloat16),
    fp16=fp16,
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    report_to=["tensorboard"],
    )


    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=(ds.get("validation") if eval_on else None),
        dataset_text_field="text",
        packing=False,
        max_seq_length=max_seq_len,
        data_collator=collator,
    )
    class PrintMetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                print(f"Step {state.global_step} - Loss: {logs.get('loss')}, Eval Loss: {logs.get('eval_loss')}")
                
    trainer.add_callback(PrintMetricsCallback)
    trainer.train(resume_from_checkpoint=resume_from)

    trainer.model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    tokenizer.save_pretrained(output_dir)

    print("Training finished. LoRA adapter saved to:", os.path.join(output_dir, "lora_adapter"))


if __name__ == "__main__":
    main()