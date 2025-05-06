# Surprisal-Calculation-for-Chinese-Words
This project calculates the surprisal of Chinese words using a pretrained language model.

The script uses a GPT-2-based Chinese model (uer/gpt2-chinese-cluecorpussmall) from Hugging Face. You can modify the model by changing the model_name in the script.

Note: It tokenizes Chinese sentences character-by-character (without using jieba segmentation). For a sentence like "他喜欢吃苹果", the script calculates the surprisal of the word "苹果" by considering each individual character ("苹" and "果").
