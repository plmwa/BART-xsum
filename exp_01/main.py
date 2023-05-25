# importライブラリ
import datetime
import random
import time
import os

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
import yaml
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
            summary_ids == 1
        ] = (
            -100
        )  # Note: the input_ids includes padding too, so replace pad tokens(zero value) with value of -100
        #なぜpaddingが1になってしまうのか


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

    def setup(self,stage=None):
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
        #https://github.com/Lightning-AI/lightning/pull/16520 より移行
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
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
            max_length=self.cfg.model.document_max_length,
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
        predicted_texts = self.predict(batch["document_ids"], batch["document_attention_mask"])
        self.validation_step_outputs.append(results["loss"])
        self.log("val/loss",results["loss"],prog_bar=True)
        return {
            "loss":results["loss"],
            "text": batch["document"],
            "summary": batch["summary"],
            "predicted_text": predicted_texts,
        }
    
    def test_step(self, batch, batch_size):
        results = self._step(batch)
        predicted_texts = self.predict(batch["document_ids"], batch["document_attention_mask"])
        self.test_step_outputs.append(results["loss"])
        self.log("test/loss",results["loss"],prog_bar=True)
        return {
            "loss":results["loss"],
            "text": batch["document"],
            "summary": batch["summary"],
            "predicted_text": predicted_texts,
        }

    #validation_epoc_endが使えなくなったらしい　v2.0.0~
    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.validation_step_outputs.clear()  # free memory
        self.log("val/loss",epoch_average)

    def on_test_epoch_end(self):
        epoch_average = torch.stack(self.test_step_outputs).mean()
        self.test_step_outputs.clear()  # free memory
        self.log("test/loss",epoch_average)

    #最適化関数
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

class CustumTrainer:
    def __init__(self,cfg):
        self.cfg = cfg
    
    def execute(self):
        current = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime(
            "%Y%m%d_%H%M%S"
        )
        MODEL_OUTPUT_DIR = "/content/drive/MyDrive/murata-lab/graduation_research/BART_xsum_practice/BART_xsum_practice_src/exp_01/outputs" + current
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

        wandb.init(
            project=self.cfg.wandb.project,
            name=current,
            config=self.cfg,
            id=current,
            save_code=True,
        )
        # データセットのダウンロード（xsum）
        xsum = load_dataset("xsum")
        train_ds = xsum["train"]
        val_ds = xsum["validation"]
        test_ds = xsum["test"]

        # DataFrame変換、よくわからん意味あるのかな？多分やんなくてもいい（Datasetクラスの記述変更は必要になる）
        train_df = pd.DataFrame(train_ds)
        val_df = pd.DataFrame(val_ds)
        test_df = pd.DataFrame(test_ds)

        print(train_df)
        # トークナイザーモデルの読み込み
        tokenizer = BartTokenizer.from_pretrained(self.cfg.model.pretrained_model_name)
        # Datasetのdocumentは2000くらい長さあるけど、今回使うBartの入力がmax1024だから1024以降の文は切り捨てる
        train_dataset = XsumDataset(
            train_df, tokenizer, document_max_length=self.cfg.model.data_module.document_max_length, summary_max_length=self.cfg.model.data_module.summary_max_length
        )
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
            batch_size=self.cfg.training.batch_size,
            document_max_token_length=self.cfg.model.document_max_length,
            summary_max_token_length=self.cfg.model.summary_max_length,
        )
        data_module.setup()

        model = CustumBart(tokenizer, self.cfg)

        
        early_stop_callback = EarlyStopping(
            self.cfg.model.early_stopping
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=MODEL_OUTPUT_DIR,
            monitor="val/loss",
            mode="min",
            filename="{epoc}",
            verbose=True,
        )

        progress_bar = RichProgressBar()

        trainer = pl.Trainer(
            max_epochs=self.cfg.model.epoch,
            accelerator="auto",
            devices="auto",
            callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
            logger=wandb_logger,
            deterministic=True,
        )

        trainer.fit(model, data_module)

        trainer.test(model, data_module)

        wandb.finish()


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    #wandbセットアップ
    wandb.login()
    wandb_logger = WandbLogger(
        log_model=False,
    )

    #sweepか普通に実行かどちらかをこのboolで選ぶ
    DO_SWEEP = True
    sweep_config = dict(
        method="random",
        metric=dict(
            goal="minimize",
            name="val/loss",
        ),
        parameters=dict(
            data_module=dict(
                parameters=dict(
                    batch_size=dict(
                        values=[1, 2, 3, 4],
                    ),
                    document_max_length=1024,  # データセットの入力テキストは21~25字
                    summary_max_length=400,
                )
            ),
            optimizer=dict(
                parameters=dict(
                    name=dict(
                        values=["AdamW", "RAdam"],
                    ),
                    lr=dict(
                        values=[1e-5, 5e-5, 9e-5, 1e-6],
                    ),
                ),
            ),
        ),
    )
    #Execute
    if DO_SWEEP:
        sweep_id = wandb.sweep(sweep_config, project=cfg.wandb.project)
        trainer = CustumTrainer(cfg)
        wandb.agent(sweep_id, trainer.execute, count=10)
    else:
        trainer = CustumTrainer(config)
        trainer.execute()



if __name__ == "__main__":
    main()
