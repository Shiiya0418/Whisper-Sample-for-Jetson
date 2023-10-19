import whisper
import sys
import torch

if len(sys.argv) < 2:
    print("Please input an jaudio file path.")
    sys.exit()
model = whisper.load_model("tiny", device="cpu")
while True:
    finetune = input("Do you use finetuning model? (yes/no):")
    if finetune == "no":
        break
    if finetune == "yes":
        model_path = "ft_whisper_best.pth"
        model.load_state_dict(torch.load(model_path))
        break
    else:
        print("Please input correctly. Try Again.")

_ = model.half()
_ = model.cuda()

for m in model.modules():
    if isinstance(m, whisper.model.LayerNorm):
        m.float()

model.eval()
with torch.inference_mode():
    result = model.transcribe(
        sys.argv[1],
        fp16=True,
        without_timestamps=True
    )
print(result["text"])
