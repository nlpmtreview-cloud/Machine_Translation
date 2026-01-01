import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import sacrebleu, evaluate
from tqdm import tqdm
import pandas as pd

# SETTINGS
MODEL_NAME = "facebook/nllb-200-distilled-600M"
DATASET_PATH = "NTREX_ta_en_benchmark/data.json"

SRC_LANG = "tam_Taml"   
TGT_LANG = "eng_Latn"  

N = 100
BATCH_SIZE = 4
MAX_LEN = 128
NUM_BEAMS = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# LOAD MODEL & TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

# LOAD DATASET
test_ds = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Loaded {len(test_ds)} samples")

test_ds = test_ds.select(range(min(N, len(test_ds))))

sources = test_ds["sourceText"]     # Tamil
references = test_ds["targetText"]  # English

# NORMALIZATION
def normalize(text):
    text = text.strip()
    return " ".join(text.split())

# BATCH TRANSLATION
def translate_batch_nllb(texts):
    tokenizer.src_lang = SRC_LANG
    all_preds = []

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    for i in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="TA → EN Inference"
    ):
        batch = texts[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=MAX_LEN,
                num_beams=NUM_BEAMS,
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        all_preds.extend(decoded)

    return all_preds

# RUN TRANSLATION
print("\nTranslating Tamil → English ...")
preds = translate_batch_nllb(sources)

preds_norm = [normalize(p) for p in preds]
refs_norm = [normalize(r) for r in references]

# SAVE RESULTS
df = pd.DataFrame({
    "Source (TA)": sources,
    "Prediction (EN)": preds_norm,
    "Reference (EN)": refs_norm
})

df.to_csv(
    "nllb_ntrex_ta2en_optimized.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Saved: nllb_ntrex_ta2en_optimized.csv")

# EVALUATION
bleu = sacrebleu.corpus_bleu(preds_norm, [refs_norm]).score
chrf = evaluate.load("chrf").compute(
    predictions=preds_norm,
    references=refs_norm
)["score"]

print("\n TA → EN RESULTS (NLLB – OPTIMIZED)")
print(f"BLEU : {bleu:.2f}")
print(f"chrF : {chrf:.2f}")
