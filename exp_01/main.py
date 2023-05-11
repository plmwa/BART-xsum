from datasets import load_datasets

all_datasets=load_datasets()
xsum = load_dataset("xsum")
print(xsum)