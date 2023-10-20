import torch
import whisper
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchaudio
from torchaudio.functional import resample
import sys

class WhisperDataCollatorWhithPadding:
    def __call__(
            sefl,
            features
            ):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch

class FinetuneDataset(Dataset):
    def __init__(
            self,
            data,
            tokenizer,
            sample_rate
            ):
        super().__init__()
        self.X = data
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        data = self.X[idx]
        x = data[0]
        waveform,sr = torchaudio.load(x, normalize=True)
        if sr != self.sample_rate:
            waveform = resample(waveform, sr, self.sample_rate) 

        mel = self.to_pad_to_mel(waveform).squeeze(0)
        text = data[1]

        #text = self.text_kana_convert(text)
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        return {"input_ids": mel, "dec_input_ids": text, "labels": labels}

    def to_pad_to_mel(self,data):
        padded_input = whisper.pad_or_trim(np.asarray(data,dtype=np.float32))
        input_ids = whisper.log_mel_spectrogram(padded_input)
        return input_ids

    #openjtalkでカナ変換を行う関数。一応残して置きます。
    #def text_kana_convert(self,text):
    #    text = pyopenjtalk.g2p(text,kana=True)
    #    return text

class EvalDataset(Dataset):
    def __init__(
            self,
            X,
            labels
            ):
        super().__init__()
        self.X = X
        self.labels = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        x = self.X[idx]
        label = self.labels[idx]
        return x, label

def load_data():
    #使用するデータの準備
    #fr = open('data/clean_metadata.csv', "r", encoding='UTF-8') #ノイズなしデータの読み込みを行うためのファイル(予備)
    fr = open('data/noise_metadata.csv', "r", encoding='UTF-8') #ノイズありデータの読み込みを行うためのファイル
    ############################################################################
    #csvの形式は以下のようなものを想定
    #my_audio_0.wav,文章
    #my_audio_1.wav,文章
    #       ・
    #       ・
    #       ・
    ############################################################################

    datalist = fr.readlines()
    train_data = []
    eval_data = []
    train_num = 100
    valid_num = 30
    for i, line1 in enumerate(datalist):
        filename = line1.split(',')[0]
        line2 = line1.split(',')[1]
        if i < train_num:                                 
            train_data.append((filename, line2))
        elif train_num <= i and i < train_num + valid_num:
            eval_data.append((filename, line2))

    print(f"train data: {len(train_data)}")
    print(f"validation data: {len(eval_data)}")

    return train_data, eval_data
