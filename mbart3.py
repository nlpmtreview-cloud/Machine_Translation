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

# LANGUAGE CODES
SRC_LANG = "en_XX"
TGT_LANG = "ta_IN"

# LOAD TEST FILES
N = 100  

with open("corpus.bcn.test.en", encoding="utf-8") as fen, \
     open("corpus.bcn.test.ta", encoding="utf-8") as fta:
    en_lines = [line.strip() for line in fen.readlines()[:N]]
    ta_lines = [line.strip() for line in fta.readlines()[:N]]

assert len(en_lines) == len(ta_lines), "Mismatch in EN/TA test files"

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

print("\n Translating English → Tamil using mBART ...")

for src, ref in tqdm(zip(en_lines, ta_lines), total=N, desc="EN→TA Inference"):
    pred = translate_mbart(src)

    sources.append(src)
    preds.append(pred)
    refs.append(ref)

# SAVE RESULTS
df = pd.DataFrame({
    "Source (EN)": sources,
    "Prediction (TA)": preds,
    "Reference (TA)": refs
})

df.to_csv("mbart_results_en2ta_100.csv", index=False, encoding="utf-8-sig")
print(" Results saved to mbart_results_en2ta_100.csv")

# EVALUATION
bleu = sacrebleu.corpus_bleu(preds, [refs]).score
chrf = evaluate.load("chrf").compute(
    predictions=preds,
    references=refs
)["score"]

print("\n mBART EN → TA EVALUATION SCORES")
print(f"BLEU : {bleu:.2f}")
print(f"chrF : {chrf:.2f}")
