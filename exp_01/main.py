#importライブラリ

#dataset
from datasets import load_dataset
#transformers
from transformers import BartTokenizer, BartModel
#torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
#pandas
import pandas as pd
#hydra
import hydra
from omegaconf import DictConfig

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
        data_row=self.data.iloc[index]
        document = data_row["document"]
        summary = data_row["summary"]

        document_encoding = self.tokenizer.encode_plus(
            document,
            max_length=self.document_max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        )

        summary_encoding = self.tokenizer.encode_plus(
            summary,
            max_length=self.document_max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        )

        return dict(
            document=document,
            document_ids=document_encoding["input_ids"].flatten(),
            document_attention_mask=document_encoding["attention_mask"].flatten(),
            summary=summary,
            summary_ids=summary_encoding["input_ids"].flatten(),
            summary_attention_mask=summary_encoding["attention_mask"].flatten(),
        )



#DataLoderの作成
class XsumDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        valid_df,
        test_df,
        tokenizer,
        batch_size,
        document_max_token_length,
        summary_max_token_length,
    ):

        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.document_max_token_length = document_max_token_length
        self.summary_max_token_length = summary_max_token_lenght
        self.tokenizer = tokenizer

    def setup(self):
        self.train_dataset = XsumDataset(
            self.train_df,
            self.tokenizer,
            self.document_max_token_length,
            self.summary_max_token_length,
        )
        self.vaild_dataset = XsumDataset(
            self.valid_df,
            self.tokenizer,
            self.document_max_token_length,
            self.summary_max_token_length,
        )
        self.test_dataset = XsumDataset(
            self.test_df,
            self.tokenizer,
            self.document_max_token_length,
            self.summary_max_token_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.vaild_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )
    


@hydra.main(config_path=".", config_name="config")

def main(cfg: DictConfig):

    #データセットのダウンロード（xsum）
    xsum = load_dataset("xsum")
    train_ds = xsum["train"]
    val_ds = xsum["validation"]
    test_ds = xsum["test"]


    #DataFrame変換、よくわからん意味あるのかな？
    train_df = pd.DataFrame(train_ds)
    val_df = pd.DataFrame(val_ds)
    test_df = pd.DataFrame(test_ds)

    print(train_df)
    #トークナイザーモデルの読み込み
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    #Datasetのdocumentは2000くらい長さあるけど、今回使うBartの入力がmax1024だから1024以降の文は切り捨てる
    train_dataset=XsumDataset(train_df,tokenizer,document_max_length=1024,summary_max_length=400)

    #トークナイズ結果確認
    for data in train_dataset:
        print("要約前文章")
        print(data["document"])
        print(data["document_ids"])
        print(data["document_attention_mask"])
        print("要約後文章")
        print(data["summary"])
        print(data["summary_ids"])
        print(data["summary_attention_mask"])
        break


    data_module = XsumDataModule(
        train_df=train_df,
        valid_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        batch_size=cfg.training.batch_size,
        document_max_token_length=cfg.model.document_max_length,
        summary_max_token_length=cfg.model.summary_max_length,
    )
    data_module.setup()

    #モデルの読み込み
    model = BartModel.from_pretrained('facebook/bart-base')

if __name__ == "__main__":
    main()