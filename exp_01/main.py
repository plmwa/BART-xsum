from datasets import load_dataset

xsum = load_dataset("xsum")
train_ds=xsum["train"]
print(len(train_ds["document"][0]))