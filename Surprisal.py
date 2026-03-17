import torch
import math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os

# ===== configuration =====
model_name = "uer/gpt2-chinese-cluecorpussmall"
input_csv = "your_file.csv"   # expects columns: 'sentence', 'target'
output_csv = "wordlevel_surprisal.csv"

# If True: estimate probability of the first token without left context (not usually recommended)
# If False: mark such cases as uncomputable (preferred for most research settings)
use_unconditional_first_token = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== load model and tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# ===== load input data =====
df = pd.read_csv(input_csv, dtype=str).fillna("")
results = []

for _, row in df.iterrows():
    sentence = row['sentence']
    target_word = row['target']

    # Locate target word at character level (used only for initial slicing)
    start_idx_char = sentence.find(target_word)
    if start_idx_char == -1:
        results.append({
            "sentence": sentence,
            "target": target_word,
            "status": "target_not_found",
            "surprisal": None
        })
        continue

    # Split into prefix (context) and full input (prefix + target)
    prefix_text = sentence[:start_idx_char]
    full_text = sentence[: start_idx_char + len(target_word)]

    # Tokenize without adding special tokens
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    # Token IDs corresponding to the target word
    target_ids = tokenizer(target_word, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).tolist()
    target_len = len(target_ids)

    # Token-level starting position of the target word
    tok_start = prefix_ids.size(1)

    # Verify alignment between full_ids and target_ids
    seq = full_ids[0].tolist()
    if tok_start + target_len > len(seq) or seq[tok_start:tok_start+target_len] != target_ids:
        # Attempt to find target_ids elsewhere in the sequence
        found_idx = -1
        for i in range(len(seq) - target_len + 1):
            if seq[i:i+target_len] == target_ids:
                found_idx = i
                break
        if found_idx >= 0:
            tok_start = found_idx
        else:
            results.append({
                "sentence": sentence,
                "target": target_word,
                "status": "token_alignment_failed",
                "surprisal": None
            })
            continue

    # Forward pass to obtain logits
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # shape: (1, sequence_length, vocab_size)

    # Compute word-level surprisal as the sum of token-level surprisals
    total_surprisal = 0.0
    uncomputable = False

    for i in range(target_len):
        token_pos = tok_start + i
        token_id = full_ids[0, token_pos].item()

        # No left context available for the first token
        if token_pos == 0:
            if use_unconditional_first_token:
                logits_prev = logits[0, 0]
            else:
                uncomputable = True
                break
        else:
            logits_prev = logits[0, token_pos - 1]

        log_probs = F.log_softmax(logits_prev, dim=-1)
        token_logprob = log_probs[token_id].item()

        # Convert from natural log to log base 2
        surprisal = - token_logprob / math.log(2)
        total_surprisal += surprisal

    if uncomputable:
        results.append({
            "sentence": sentence,
            "target": target_word,
            "status": "uncomputable_no_context",
            "surprisal": None,
            "tok_start": tok_start,
            "target_token_ids": ",".join(map(str, target_ids))
        })
    else:
        results.append({
            "sentence": sentence,
            "target": target_word,
            "status": "ok",
            "surprisal": round(total_surprisal, 6),
            "tok_start": tok_start,
            "target_token_ids": ",".join(map(str, target_ids))
        })

# ===== save results =====
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
pd.DataFrame(results).to_csv(output_csv, index=False)

print(f"Done. Results saved to {output_csv}")
