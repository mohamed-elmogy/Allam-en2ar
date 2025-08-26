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
MODEL_PATH = "./allam-en2ar-lora"              # path to fine-tuned model
DATA_PATH = "test_set.csv"                     # test set CSV with 'en' and 'ar' or 'en_clean' and 'ar_clean'
OUTPUT_FILE = "eval_results.csv"               # where to save results table
MAX_SAMPLES = 200                              # number of samples to evaluate
MAX_NEW_TOKENS = 128                           # max tokens for generation

# -----------------------------
# Generate translations
# -----------------------------
def generate_translations(model, tokenizer, sources, max_new_tokens=128):
    preds = []
    for src in sources:
        prompt = f"Translate the following sentence from English to Arabic.\n\nEnglish: {src}\nArabic:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only Arabic part
        if "Arabic:" in text:
            text = text.split("Arabic:")[-1].strip()
        preds.append(text)
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

    # Load test data
    df = pd.read_csv(DATA_PATH)
    df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=42)
    sources = df["en"].tolist() if "en" in df else df["en_clean"].tolist()
    refs = df["ar"].tolist() if "ar" in df else df["ar_clean"].tolist()

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
    eed = evaluate.load("eed")
    results["EED"] = eed.compute(predictions=preds, references=refs)["eed"]

    # LLM-as-a-Judge (LLMG) – optional
    try:
        llmg = evaluate.load("llmg")
        llmg_out = llmg.compute(predictions=preds, references=refs)
        results["LLMG"] = llmg_out["score"]
    except Exception:
        results["LLMG"] = "Not Available (need API access)"

    # -----------------------------
    # Save results
    # -----------------------------
    results_df = pd.DataFrame([results])
    results_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print("Evaluation results saved to", OUTPUT_FILE)
    print(results_df)


if __name__ == "__main__":
    main()
