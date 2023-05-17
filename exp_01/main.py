#データセットのダウンロード（xsum）
from datasets import load_dataset

xsum = load_dataset("xsum")
train_ds=xsum["train"]
print(xsum)
sample_text = xsum["train"]["document"][:2000]
print(len(sample_text))

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
string="The US are a country. The organizatino."
sent_tokenize(string)