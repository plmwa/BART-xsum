from datasets import list_datasets

all_datasets=list_datasets()
xsum = load_dataset("xsum")
print(xsum)