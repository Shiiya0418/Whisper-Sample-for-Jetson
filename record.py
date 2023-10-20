import whisper
import pyaudio
import numpy as np
import wave
import struct
import os
import sys
from glob import glob

def record(idx, sr, framesize, t):
    pa = pyaudio.PyAudio()
    data = []
    dt = 1 / sr
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, input_device_index=idx, frames_per_buffer=framesize)
    for i in range(int(((t / dt) / framesize))):
        frame = stream.read(framesize)
        data.append(frame)

    stream.stop_stream()
    stream.close()
    pa.terminate()
    data = b"".join(data)
    data = np.frombuffer(data,dtype="int16")
    return data

sr = 48000
framesize = 1024
idx = 11
t = 4

path = "./*.wav"
paths = glob(path)
wav_id = len(paths) // 2

print("recording...")
data = record(idx, sr, framesize, t)
print("finish")
bi_wave = struct.pack("h" * len(data), *data)

w = wave.Wave_write(f"my_audio_{wav_id}.wav")
p = (1, 2, sr, len(bi_wave), 'NONE', 'not compressed')
w.setparams(p)
w.writeframes(bi_wave)
w.close()