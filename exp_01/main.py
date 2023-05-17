#データセットのダウンロード（xsum）
from datasets import load_dataset

xsum = load_dataset("xsum")
train_ds=xsum["train"]
print(xsum)
sample_text = xsum["train"]["document"][:2000]
print(sample_text)