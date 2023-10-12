import wave
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Please input an audio file.")
    sys.exit()

else:
    file_name = sys.argv[1]

    wave_file = wave.open(file_name,"r")
    x = wave_file.readframes(wave_file.getnframes())
    x = np.frombuffer(x,dtype="int16")
    y = x + (np.random.randn(len(x)) * 5000).astype(np.int16)

    split_file_name = file_name.split(".")

    w = wave.Wave_write(f"{split_file_name[0]}_noise.{split_file_name[1]}")
    w.setparams(wave_file.getparams())
    w.writeframes(y)
    w.close()

