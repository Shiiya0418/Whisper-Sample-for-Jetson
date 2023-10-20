import torch
#import librosa
import whisper
#import numpy as np
import torch.nn as nn
import evaluate
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning import LightningModule

from loader import *
import sys

class WhisperModelModule(LightningModule):
    def __init__(
            self, 
            cfg=None, 
            model_name="base", 
            lang="ja", 
            train_dataset=[], 
            eval_dataset=[],
            train_data_num=1,
            valid_data_num=1,
            save_name = None
            ) -> None:
        super().__init__()
        # モデルやトークナイザーの設定です。
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.save_name = save_name
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="ja", task=self.options.task)

        # エンコーダによる音響特徴量の抽出部分の学習は行いません。
        #for p in self.model.encoder.parameters():
        #    p.requires_grad = False
        
        # Discussionに書かれてましたが、CrossEntropyを使っているそうです
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # WERとCERを計算する関数です。
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.__train_dataset = train_dataset # List[(audioのID, audioのパス, audioのテキスト)]
        self.__eval_dataset = eval_dataset # List[(audioのID, audioのパス, audioのテキスト)]
        self.max_loss = sys.maxsize
        self.train_data_num = train_data_num
        self.valid_data_num = valid_data_num
        self.train_loss = []
        self.train_batch_loss = []
        self.valid_loss = []
        self.valid_batch_loss = []
        self.batch_cer = []
        self.cer = []
        self.batch_wer = []
        self.wer = []
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids) # ここは学習しない

        out = self.model.decoder(dec_input_ids, audio_features) # デコーダのみ学習
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.train_batch_loss.append(loss.item())
        if len(self.train_batch_loss) == self.train_data_num // input_ids.shape[0]:
            self.train_loss.append(np.mean(self.train_batch_loss))
            self.train_batch_loss = []
        return loss
    
    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()


        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        # 以下、トークンをカナ(テキスト)に変換
        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o)) 
            l_list.append(self.tokenizer.decode(l))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        self.valid_batch_loss.append(loss.item())
        self.batch_cer.append(cer)
        self.batch_wer.append(wer)
        if len(self.valid_batch_loss) == self.valid_data_num // input_ids.shape[0]:
            self.valid_loss.append(np.mean(self.valid_batch_loss))
            self.cer.append(np.mean(self.batch_cer))
            self.wer.append(np.mean(self.batch_wer))
            self.valid_batch_loss = []
            self.batch_cer = []
            self.batch_wer = []


        if loss.item() < self.max_loss:
            self.max_loss = loss.item()
            torch.save(self.model.state_dict(), self.save_name)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=self.cfg.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        """初期設定"""

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
    
    def train_dataloader(self):
        """訓練データローダーを作成する"""
        dataset = FinetuneDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset,
                          batch_size=self.cfg.batch_size, 
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )

    def val_dataloader(self):
        """評価データローダーを作成する"""
        print(self.cfg)
        dataset = FinetuneDataset(self.__eval_dataset, self.tokenizer, self.cfg.sample_rate)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=1, 
                          num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding()
                          )
