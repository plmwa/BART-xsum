#importライブラリ

#データセット
from datasets import load_dataset
#transformers
from transformers import BartTokenizer, BartModel
#torch
from torch.utils.data import DataLoader, Dataset

#データセットのダウンロード（xsum）


xsum = load_dataset("xsum")
train_ds = xsum["train"]

#Xsumのオブジェクト
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
        data_row=self.data[index]
        document = data_row["document"]
        summary = data_row["summary"]

        document_encoding = self.tokenizer.encode_plus(
            document,
            max_length=self.document_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        summary_encoding = self.tokenizer.encode_plus(
            summary,
            max_length=self.document_max_length,
            padding="max_length",
            truncation=True,
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
i=1
for data in train_dataset:
    print("要約前文章")
    print(data["document"])
    print(data["document_ids"])
    print("要約後文章")
    print(data["summary"])
    print(data["summary_ids"])
    if i==2: 
        break
    i+=1
