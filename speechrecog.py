import whisper
import sys

if len(sys.argv) < 2:
    print("Please input an jaudio file path.")
    sys.exit()
model = whisper.load_model("tiny", device="cpu")
_ = model.half()
_ = model.cuda()

for m in model.modules:
    if isinstance(m, whisper.model.LayerNorm):
        m.float()

result = model.transcribe(
    sys.argv[1],
    fp16=True,
    without_timestamps=True
)
print(result["text"])
