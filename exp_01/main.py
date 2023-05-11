from datasets import load_dataset

xsum = load_dataset("xsum")
train_ds=xsum["train"]
print(train_ds.features)