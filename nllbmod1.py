import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd
import sacrebleu, evaluate
from peft import LoraConfig, get_peft_model

# =========================
# SETTINGS
# =========================
model_name = "facebook/nllb-200-distilled-600M"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# LOAD TOKENIZER & MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# =========================
# APPLY LoRA (architecture preserved)
# =========================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

model.eval()   # 🔴 VERY IMPORTANT for fast inference

# =========================
# LOAD TEST DATA
# =========================
N = 100
with open("corpus.bcn.test.en", encoding="utf-8") as fen, \
     open("corpus.bcn.test.ta", encoding="utf-8") as fta:
    en_lines = [line.strip() for line in fen.readlines()[:N]]
    ta_lines = [line.strip() for line in fta.readlines()[:N]]

assert len(en_lines) == len(ta_lines)

# =========================
# FAST BATCH TRANSLATION
# =========================
def translate_batch_nllb(texts, batch_size=4):
    tokenizer.src_lang = "eng_Latn"
    all_preds = []

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="EN → TA Inference"
    ):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        forced_bos_token_id = tokenizer.convert_tokens_to_ids("tam_Taml")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=4,          # ⚡ reduced beams
                early_stopping=True  # ⚡ stops when done
            )

        decoded = tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        all_preds.extend(decoded)

    return all_preds

# =========================
# RUN TRANSLATION
# =========================
print("\nTranslating English → Tamil ...")
preds = translate_batch_nllb(en_lines)

# =========================
# TEXT NORMALIZATION
# =========================
def normalize_text(text):
    text = text.strip()
    text = text.replace("।", ".")
    return " ".join(text.split())

preds_norm = [normalize_text(p) for p in preds]
refs_norm = [normalize_text(r) for r in ta_lines]

# =========================
# SAVE OUTPUT
# =========================
df = pd.DataFrame({
    "Source (EN)": en_lines,
    "Prediction (TA)": preds_norm,
    "Reference (TA)": refs_norm
})

df.to_csv(
    "nllb_results_en2ta_fast.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Saved: nllb_results_en2ta_fast.csv")

# =========================
# EVALUATION
# =========================
bleu = sacrebleu.corpus_bleu(preds_norm, [refs_norm]).score
chrf = evaluate.load("chrf").compute(
    predictions=preds_norm,
    references=refs_norm
)["score"]

print("\nEN → TA EVALUATION (FAST MODE)")
print(f"BLEU : {bleu:.2f}")
print(f"chrF : {chrf:.2f}")
