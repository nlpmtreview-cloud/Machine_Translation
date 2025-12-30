import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from datasets import load_dataset
import sacrebleu, evaluate
from tqdm import tqdm
import pandas as pd

# SETTINGS
model_name = "facebook/mbart-large-50-many-to-many-mmt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load model and tokenizer
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

# Load JSON dataset
dataset_path = "NTREX_ta_en_benchmark/data.json"
test_ds = load_dataset("json", data_files=dataset_path, split="train")
print(f"Loaded {len(test_ds)} examples")

N = 100  

# TRANSLATION FUNCTION
def translate_mbart(text, src_lang="en_XX", tgt_lang="ta_IN"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    generated_ids = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=128
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# RUN TRANSLATION & EVALUATE
def run_translation(src_col, ref_col, src_lang, tgt_lang, csv_name):
    preds, refs, sources = [], [], []
    print(f"\nTranslating {src_lang} → {tgt_lang} ...")

    for i in tqdm(range(min(N, len(test_ds))), desc=f"{src_lang}->{tgt_lang} Inference"):
        src = test_ds[i][src_col]
        ref = test_ds[i][ref_col]
        pred = translate_mbart(src, src_lang=src_lang, tgt_lang=tgt_lang)

        preds.append(pred)
        refs.append(ref)
        sources.append(src)

    # Save CSV
    df = pd.DataFrame({
        "Source": sources,
        "Prediction": preds,
        "Reference": refs
    })
    df.to_csv(csv_name, index=False, encoding="utf-8-sig")
    print(f"Saved results to {csv_name}")

    # Evaluate BLEU and chrF
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = evaluate.load("chrf").compute(predictions=preds, references=refs)["score"]

    print(f"\n{src_lang} → {tgt_lang} SCORES")
    print(f"BLEU : {bleu:.2f}")
    print(f"chrF : {chrf:.2f}")

# English → Tamil ONLY
run_translation(
    src_col="targetText",  
    ref_col="sourceText", 
    src_lang="en_XX",
    tgt_lang="ta_IN",
    csv_name="mbart_results_en2ta.csv"
)
