import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import sacrebleu, evaluate
from tqdm import tqdm
import pandas as pd

# LOAD MODEL
model_name = "facebook/nllb-200-distilled-600M"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# LOAD DATASET
test_ds = load_from_disk("test_ds")
N = 100  

# ORIGINAL TRANSLATION FUNCTION
def translate_nllb(text, src_lang="tam_Taml", tgt_lang="eng_Latn"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    out = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=128
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def run_translation(src_col, ref_col, src_lang, tgt_lang, csv_name):
    preds, refs, sources = [], [], []
    print(f"\nTranslating {src_lang} → {tgt_lang} ...")

    for i in tqdm(range(N), desc=f"{src_lang}->{tgt_lang} Inference"):
        src = test_ds[i][src_col]
        ref = test_ds[i][ref_col]
        pred = translate_nllb(src, src_lang=src_lang, tgt_lang=tgt_lang)

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

    # Evaluate
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = evaluate.load("chrf").compute(predictions=preds, references=refs)["score"]
    meteor = evaluate.load("meteor").compute(predictions=preds, references=refs)["meteor"]

    print(f"\n{src_lang} → {tgt_lang} SCORES")
    print(f"BLEU   : {bleu:.2f}")
    print(f"chrF   : {chrf:.2f}")
    print(f"METEOR : {meteor:.4f}")

#  Tamil → English
run_translation("ta", "en", "tam_Taml", "eng_Latn", "nllb_results_ta2en.csv")

#  English → Tamil
run_translation("en", "ta", "eng_Latn", "tam_Taml", "nllb_results_en2ta.csv")
