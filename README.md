# Surprisal-Calculation-for-Chinese-Words

This project calculates the surprisal of Chinese words using a pretrained language model.

The script uses a GPT-2-based Chinese model (`uer/gpt2-chinese-cluecorpussmall`) from Hugging Face. You can modify the model by changing the `model_name` in the script.

Surprisal is computed based on token-level probabilities from the model. For a given target word, its surprisal is defined as the sum of the surprisals of its constituent tokens.

Note: Chinese text is tokenized using the model’s tokenizer (typically at the character level). For example, in the sentence "他喜欢吃苹果", the word "苹果" is typically split into two tokens ("苹" and "果"), and its surprisal is computed as the sum of their surprisals.

The current implementation uses token-level alignment to ensure that probability estimates are consistent with the autoregressive nature of the model.
