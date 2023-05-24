# importライブラリ
import datetime
import random
import time
# dataset
from datasets import load_dataset
# transformers
from transformers import BartTokenizer, BartModel,BartForConditionalGeneration
# torch
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint,RichProgressBar
# pandas
import pandas as pd
# hydra
import hydra
from omegaconf import DictConfig
#wandb
import wandb
from pytorch_lightning.loggers import WandbLogger
# Xsumのオブジェクト
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

# Dataset読み込みクラス


class XsumDataset(Dataset):
    def __init__(self, data, tokenizer, document_max_length, summary_max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.document_max_length = document_max_length
        self.summary_max_length = summary_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
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
        summary_ids = summary_encoding["input_ids"]
        summary_ids[
            summary_ids == 0
        ] = (
            -100
        )  # Note: the input_ids includes padding too, so replace pad tokens(zero value) with value of -100


        return dict(
            document=document,
            document_ids=document_encoding["input_ids"].flatten(),
            document_attention_mask=document_encoding["attention_mask"].flatten(
            ),
            summary=summary,
            summary_ids=summary_encoding["input_ids"].flatten(),
            summary_attention_mask=summary_encoding["attention_mask"].flatten(
            ),
        )


# DataLoderの作成
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
        self.summary_max_token_length = summary_max_token_length
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


# モデル定義
class CustumBart(pl.LightningModule):
    def __init__(
        self,
        tokenizer,
        cfg,
        pretrained_model="facebook/bart-base",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.model=BartForConditionalGeneration.from_pretrained(pretrained_model)
    
    def forward(
        self,
        document_ids,
        document_attention_mask=None,
        summary_ids=None,
        summary_attention_mask=None,
    ):
        output = self.model(
            input_ids=document_ids,
            attention_mask=document_attention_mask,
            labels=summary_ids,
            decoder_attention_mask=summary_attention_mask,
        )
        return output.loss,output.logits
    
    def predict(self,
                document_ids,
                document_attention_mask,
    ):
        output = self.model.generate(
            document_ids,
            attention_mask=document_attention_mask,
            max_length=self.cfg.model.max_length,
            num_beams=1,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )
        return[
            self.tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for ids in output
        ]
    
    def _step(self,batch,return_text=False):
        loss,logits = self(
            document_ids=batch["document_ids"],
            document_attention_mask=batch["document_attention_mask"],
            summary_ids=batch["summary_ids"],
            summary_attention_mask=batch["summary_attention_mask"],
        )
        return{
            "loss":loss,
            "logits":logits,
            "document_ids":batch["document_ids"],
            "summary_ids":batch["summary_ids"],
        }
    
    def training_step(self,batch,batch_size):
        results = self._step(batch)
        self.log("train/loss",results["loss"],prog_bar=True)
        return results
    
    def validation_step(self, batch, batch_size):
        results = self._step(batch)
        predicted_texts = self.predict(batch["text_ids"], batch["text_attention_mask"])
        self.log("val/loss",results["loss"],prog_bar=True)
        return {
            "loss":results["loss"],
            "text": batch["text"],
            "summary": batch["summary"],
            "predicted_text": predicted_texts,
        }
    
    def test_step(self, batch, batch_size):
        results = self._step(batch)
        predicted_texts = self.predict(batch["text_ids"], batch["text_attention_mask"])
        self.log("test/loss",results["loss"],prog_bar=True)
        return {
            "loss":results["loss"],
            "text": batch["text"],
            "summary": batch["summary"],
            "predicted_text": predicted_texts,
        }

    def _epoch_end(self, outputs, mode):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(f"{mode}/loss", avg_loss)

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, mode="val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, mode="test")




@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):

    # データセットのダウンロード（xsum）
    xsum = load_dataset("xsum")
    train_ds = xsum["train"]
    val_ds = xsum["validation"]
    test_ds = xsum["test"]

    # DataFrame変換、よくわからん意味あるのかな？多分やんなくてもいい（Datasetクラスの記述は必要になる）
    train_df = pd.DataFrame(train_ds)
    val_df = pd.DataFrame(val_ds)
    test_df = pd.DataFrame(test_ds)

    print(train_df)
    # トークナイザーモデルの読み込み
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # Datasetのdocumentは2000くらい長さあるけど、今回使うBartの入力がmax1024だから1024以降の文は切り捨てる
    train_dataset = XsumDataset(
        train_df, tokenizer, document_max_length=1024, summary_max_length=400)

    # トークナイズ結果確認
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

    #Trainer    
    wandb.login()
    model = CustumBart(tokenizer, cfg)
    current = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime(
        "%Y%m%d_%H%M%S"
    )
    wandb.init(
        project=cfg.wandb.project,
        name=current,
        config=cfg,
        id=current,
        save_code=True,
    )
    #ここらへんはあとでhydraに
    early_stopping=dict(
        monitor="val/loss",
        patience=3,
        mode="min",
        min_delta=0.02,
    )
    early_stop_callback = EarlyStopping(
        early_stopping,
    )
    wandb_logger = WandbLogger(
        log_model=False,
    )
    wandb_logger.watch(model, log="all")


    MODEL_OUTPUT_DIR = "/content/drive/MyDrive/MurataLab/summary/models/" + current
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_OUTPUT_DIR,
        monitor="val/loss",
        mode="min",
        filename="{epoch}",
        verbose=True,
    )

    progress_bar = RichProgressBar()

    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
        logger=wandb_logger,
        deterministic=True,
    )

    trainer.fit(model, data_module)

    trainer.test(model, data_module)



if __name__ == "__main__":
    main()
