#!/usr/bin/env python
"""
Evaluation script for EN→AR translation model.
Computes BLEU, METEOR, BERTScore, EED, and LLM-as-a-Judge (LLMG).
Results are saved in a CSV table.
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate

# -----------------------------
# Static Configuration
# -----------------------------
MODEL_PATH = "./allam-en2ar-lora-merged_v5"              # path to fine-tuned model
DATA_PATH = "D:/Allam-en2ar-main/Data/test_data.csv"                     # test set CSV with 'en' and 'ar' or 'en_clean' and 'ar_clean'
OUTPUT_FILE = "eval_results_v5.csv"               # where to save results table
MAX_SAMPLES = 1000                              # number of samples to evaluate
MAX_NEW_TOKENS = 128                           # max tokens for generation

# -----------------------------
# Generate translations
# -----------------------------
import re

VALID_TOKENS = {"<en>", "<ar>", "<eos>"}

def strip_invalid_special_tokens(text, valid_tokens=VALID_TOKENS):
    # Find all <...> tokens
    found_tokens = set(re.findall(r"<[^>]+>", text))
    for token in found_tokens:
        if token not in valid_tokens:
            text = text.replace(token, "")
    return text.strip()

def clean_output(text):
    # Remove invalid special tokens
    text = strip_invalid_special_tokens(text)
    # Optional: remove leftover malformed tokens like "<e"
    text = re.sub(r"<[^>]*", "", text) 
    text = re.sub(r"none", "", text) # Catches things like <e
    return text.strip()

def generate_translations(model, tokenizer, sources, max_new_tokens=128):
    preds = []
    for src in sources:
        prompt = f"Translate the following sentence from English to Arabic.\n\nEnglish: {src}\nArabic:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print("src: ",src)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                do_sample=False,         # enable sampling
                top_p=0.9,              # nucleus sampling
                temperature=0.7,        # less repetitive
                repetition_penalty=1.2,  # discourages loops
                no_repeat_ngram_size=3, 
                eos_token_id=tokenizer.eos_token_id 
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    # Extract only Arabic part
    if "\narabic:" in text:
        text = text.split("\narabic:")[-1].strip()
    preds.append(clean_output(text))
    return preds

# -----------------------------
# Main
# -----------------------------
def main():
    # Load model + tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    source = 'well we closed your joint account we added hishams new address we created an account for ola for the child support expenses'
    prompt = f"Translate the following sentence from English to Arabic.\n\nEnglish: {source}\nArabic:"
    preds = []
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            do_sample=False,         # enable sampling
            top_p=0.9,              # nucleus sampling
            temperature=0.7,        # less repetitive
            repetition_penalty=1.2,  # discourages loops
            no_repeat_ngram_size=3, 
            eos_token_id=tokenizer.eos_token_id,
            num_beams=5,
            early_stopping=True
             
        )
    # Load test data
    df = pd.read_csv(DATA_PATH)
    #df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=42)
    sources = df["en_clean"].tolist()
    refs = df["ar_clean"].tolist()
    # Generate predictions
    preds = generate_translations(model, tokenizer, sources, MAX_NEW_TOKENS)

    # -----------------------------
    # Metrics
    # -----------------------------
    results = {}
    # BLEU
    bleu = evaluate.load("sacrebleu")
    results["BLEU"] = bleu.compute(predictions=preds, references=[[r] for r in refs])["score"]

    # METEOR
    meteor = evaluate.load("meteor")
    results["METEOR"] = meteor.compute(predictions=preds, references=refs)["meteor"]

    # BERTScore
    bertscore = evaluate.load("bertscore")
    bert_out = bertscore.compute(predictions=preds, references=refs, lang="ar")
    results["BERTScore_F1"] = sum(bert_out["f1"]) / len(bert_out["f1"])

    # EED
    # -----------------------------
    # Save results
    # -----------------------------
    results_df = pd.DataFrame([results])
    results_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print("Evaluation results saved to", OUTPUT_FILE)
    print(results_df)


if __name__ == "__main__":
    main()
