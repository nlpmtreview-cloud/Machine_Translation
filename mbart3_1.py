import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import sacrebleu, evaluate
from tqdm import tqdm
import pandas as pd

# LOAD MODEL
model_name = "facebook/mbart-large-50-many-to-many-mmt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

# LANGUAGE CODES (Tamil → English)
SRC_LANG = "ta_IN"
TGT_LANG = "en_XX"

# LOAD TEST FILES
N = 100 

with open("corpus.bcn.test.ta", encoding="utf-8") as fta, \
     open("corpus.bcn.test.en", encoding="utf-8") as fen:
    ta_lines = [line.strip() for line in fta.readlines()[:N]]
    en_lines = [line.strip() for line in fen.readlines()[:N]]

assert len(ta_lines) == len(en_lines), "Mismatch in TA/EN test files"

# TRANSLATION FUNCTION
def translate_mbart(text):
    tokenizer.src_lang = SRC_LANG

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    forced_bos_token_id = tokenizer.lang_code_to_id[TGT_LANG]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# RUN TRANSLATION
preds, refs, sources = [], [], []

print("\n Translating Tamil → English using mBART ...")

for src, ref in tqdm(zip(ta_lines, en_lines), total=N, desc="TA→EN Inference"):
    pred = translate_mbart(src)

    sources.append(src)
    preds.append(pred)
    refs.append(ref)

# SAVE RESULTS
df = pd.DataFrame({
    "Source (TA)": sources,
    "Prediction (EN)": preds,
    "Reference (EN)": refs
})

df.to_csv("mbart_results_ta2en_100.csv", index=False, encoding="utf-8-sig")
print(" Results saved to mbart_results_ta2en_100.csv")

# EVALUATION
bleu = sacrebleu.corpus_bleu(preds, [refs]).score
chrf = evaluate.load("chrf").compute(
    predictions=preds,
    references=refs
)["score"]

print("\n mBART TA → EN EVALUATION SCORES")
print(f"BLEU : {bleu:.2f}")
print(f"chrF : {chrf:.2f}")
