import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from datasets import load_from_disk
import sacrebleu, evaluate
from tqdm import tqdm
import pandas as pd

# LOAD MBART MODEL
model_name = "facebook/mbart-large-50-many-to-many-mmt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

#  LOAD WIKIMATRIX DATASET
test_ds = load_from_disk("wikimatrix_test_ds")

N = 100
N = min(N, len(test_ds))
print("Evaluation samples:", N)

# TRANSLATION FUNCTION
def translate_mbart(text, src_lang="ta_IN", tgt_lang="en_XX"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

    out = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=128
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# RUN TRANSLATION + EVALUATION
def run_translation(src_col, ref_col, src_lang, tgt_lang, csv_name):
    preds, refs, sources = [], [], []
    print(f"\nTranslating {src_lang} → {tgt_lang} ...")

    for i in tqdm(range(N), desc=f"{src_lang}->{tgt_lang} Inference"):
        src = test_ds[i][src_col]
        ref = test_ds[i][ref_col]
        pred = translate_mbart(src, src_lang=src_lang, tgt_lang=tgt_lang)

        preds.append(pred)
        refs.append(ref)
        sources.append(src)

    # SAVE CSV
    df = pd.DataFrame({
        "Source": sources,
        "Prediction": preds,
        "Reference": refs
    })
    df.to_csv(csv_name, index=False, encoding="utf-8-sig")
    print("Saved:", csv_name)

    # EVALUATION
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = evaluate.load("chrf").compute(predictions=preds, references=refs)["score"]
    meteor = evaluate.load("meteor").compute(predictions=preds, references=refs)["meteor"]

    print(f"\n{src_lang} → {tgt_lang} SCORES")
    print(f"BLEU   : {bleu:.2f}")
    print(f"chrF   : {chrf:.2f}")
    print(f"METEOR : {meteor:.4f}")

# Tamil → English
run_translation("ta", "en", "ta_IN", "en_XX", "wikimatrix_mbart_ta2en.csv")

# English → Tamil
run_translation("en", "ta", "en_XX", "ta_IN", "wikimatrix_mbart_en2ta.csv")
