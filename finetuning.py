import torch
#import librosa
import whisper
import numpy as np
import torch.nn as nn
from torch import optim
from pytorch_lightning import Trainer

from loader import load_data
from whisper_mode_module import WhisperModelModule

class Config:
    def __init__(self):
        #self.learning_rate = 0.0001
        self.learning_rate = 0.00001
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.warmup_steps = 2
        #self.batch_size = 16
        self.batch_size = 16
        self.num_worker = 2
        self.num_train_epochs = 30
        self.gradient_accumulation_steps = 1
        self.sample_rate = 16000

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
    #save_name = "ft_whisper_ROHAN4600_1000.pth"
    save_name = "ft_whisper_best.pth"
    lang = "ja"
    cfg = Config()
    train_data,eval_data = load_data()
    model = WhisperModelModule(cfg,model_name,lang,train_data,eval_data,save_name)
    trainer = Trainer(
        precision=16,
        accelerator="gpu",
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps
    )
    trainer.fit(model)

if __name__ == "__main__":
    main()
