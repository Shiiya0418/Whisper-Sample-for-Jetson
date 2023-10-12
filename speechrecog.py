import whisper
import sys

if len(sys.argv) < 2:
    print("Please input an jaudio file path.")
    sys.exit()
model = whisper.load_model("base").cuda()
result = model.transcribe(sys.argv[1])
print(result["text"])
