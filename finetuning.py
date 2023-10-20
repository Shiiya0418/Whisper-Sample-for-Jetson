import torch
#import librosa
import whisper
import numpy as np
import torch.nn as nn
from torch import optim
from pytorch_lightning import Trainer

from loader import load_data
from whisper_mode_module import WhisperModelModule
import os
import shutil

import matplotlib.pyplot as plt
import csv

class Config:
    def __init__(self,
                 learning_rate=0.0001,
                 weight_decay=0.01,
                 adam_epsilon=1e-8,
                 warmup_steps=2,
                 batch_size=8,
                 num_worker=2,
                 num_train_epochs=30,
                 gradient_accumulation_steps=1,
                 sample_rate=16000):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.num_train_epochs = num_train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.sample_rate = sample_rate

    def __str__(self):
        text = "#" * 30 + "\n"
        text += f"learning rate:{self.learning_rate}\n"
        text += f"weight decay:{self.weight_decay}\n"
        text += f"adam epsilon:{self.adam_epsilon}\n"
        text += f"warmup steps:{self.warmup_steps}\n"
        text += f"batch size:{self.batch_size}\n"
        text += f"num worker:{self.num_worker}\n"
        text += f"num train epochs:{self.num_train_epochs}\n"
        text += f"gradient accumulation steps:{self.gradient_accumulation_steps}\n"
        text += f"sample_rate:{self.sample_rate}\n"
        text += "#" * 30 + "\n"

        return text

def main():
    model_name = "tiny"
    save_name = "ft_whisper_best.pth"
    lang = "ja"
    cfg = Config()
    train_data,eval_data = load_data()
    train_data_num = len(train_data)
    eval_data_num = len(eval_data)
    model = WhisperModelModule(
        cfg,
        model_name,
        lang,train_data,
        eval_data,
        train_data_num,
        eval_data_num,
        save_name
    )
    trainer = Trainer(
        precision=16,
        accelerator="gpu",
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps
    )
    trainer.fit(model)
    plt.plot([x for x in range(len(model.train_loss))], model.train_loss, label = "train")
    plt.plot([x for x in range(len(model.valid_loss))], model.valid_loss, label = "valid")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("loss_curve.png")

    with open("train_log.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss", "valid_cer", "valid_wer"])
        for i in range(cfg.num_train_epochs):
            info = [f"{i + 1}:", model.train_loss[i], model.valid_loss[i], model.cer[i], model.wer[i]]
            writer.writerow(info)

if __name__ == "__main__":
    main()
    if os.path.exists("lightning_logs"):
        shutil.rmtree("lightning_logs")
