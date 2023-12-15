from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperProcessor
import torch
import torchaudio
from torchaudio.functional import resample

def transcribe(path):
    raw_audio, sr = torchaudio.load(path)
    sr = 16000
    #raw_audio = resample(waveform=raw_audio, orig_freq=sr, new_freq=16000)
    raw_audio = raw_audio.squeeze(dim = 0)
    # feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").cuda()
    input_features = processor(raw_audio, sampling_rate=sr, return_tensors="pt").input_features
    with torch.no_grad():
        generated_ids = model.generate(input_features.to("cuda"), language="<|ja|>", task="transcribe")[0]
        transcription = processor.decode(generated_ids)
    return processor.tokenizer._normalize(transcription)

def main():
    path = "data/data/clean/my_audio_0.wav"
    result = transcribe(path)
    print("")
    print(result)
    
if __name__ == "__main__":
    main()
