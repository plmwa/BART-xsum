#importライブラリ

#データセット
from datasets import load_dataset
#transformers
from transformers import BartTokenizer, BartModel
#torch
from torch.utils.data import DataLoader, Dataset

#データセットのダウンロード（xsum）


xsum = load_dataset("xsum")

"""
DatasetDict({
    train: Dataset({
        features: ['document', 'summary', 'id'],
        num_rows: 204045
    })
    validation: Dataset({
        features: ['document', 'summary', 'id'],
        num_rows: 11332
    })
    test: Dataset({
        features: ['document', 'summary', 'id'],
        num_rows: 11334
    })
})
"""

class XsumDataset(Dataset):
    def __init__(self,data,tokenizer,document_max_length,summary_max_length):
        self.data=data
        self.tokenizer=tokenizer
        self.document_max_length=document_max_length
        self.summary_max_length=summary_max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        data_row=data[index]
        document = data_row[self.document]
        summary = data_row[self.summary]

        document_encoding = self.tokenizer.encode_plus(
            document,
            max_length=self.document_max_length,
            padding="max_length",
            return_tensors="pt",
        )

        summary_encoding = self.tokenizer.encode_plus(
            summary,
            max_length=self.document_max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return dict(
            document=document,
            document_ids=document_encoding["input_ids"].flatten(),
            summary=summary,
            summary_ids=summary_encoding["input_ids"].flatten(),
        )







#トークナイザーモデルの読み込み
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

train_dataset=XsumDataset(train_ds,tokenizer,document_max_length=2000,summary_max_length=400)

#トークナイズ結果確認
for data in train_dataset:
    print("document")
    print(data["document"])
    print(data["document_ids"])
    print(data["summary"])
    print(data["summary_ids"])

    break
