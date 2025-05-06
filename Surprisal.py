import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# Load autoregressive Chinese model
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Load input CSV
df = pd.read_csv("test.csv")  # expects columns: 'sentence', 'target'

results = []

for _, row in df.iterrows():
    sentence = row['sentence']
    target_word = row['target']

    # Tokenize sentence at character level (no segmentation required)
    tokens = list(sentence)  # Break sentence into individual characters
    
    # Find the position of the target word within the tokens (if it's multi-character)
    start_idx = sentence.find(target_word)
    if start_idx == -1:
        print(f"Target word '{target_word}' not found in: {sentence}")
        continue

    # Tokenize the sentence up to the target word
    prefix = sentence[:start_idx]
    full_input = sentence[:start_idx+len(target_word)]

    # Tokenize the full input
    input_ids = tokenizer(full_input, return_tensors="pt")["input_ids"]

    # Get model logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Find the token IDs for the characters in the target word
    target_tokens = tokenizer.tokenize(target_word)
    target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)

    # Sum the probabilities for each target token
    surprisal_sum = 0
    for i, tok_id in enumerate(target_token_ids):
        # Get the probability distribution for the current token
        prob = torch.softmax(logits[0, start_idx + i], dim=-1)
        token_prob = prob[tok_id].item()
        
        # Calculate surprisal (negative log of probability)
        surprisal = -math.log2(token_prob + 1e-12)
        surprisal_sum += surprisal

    # Store results
    results.append({
        "sentence": sentence,
        "target": target_word,
        "surprisal": round(surprisal_sum, 3)
    })

# Output to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("output.csv", index=False)
print("Done! Results saved to surprisal_output.csv")
