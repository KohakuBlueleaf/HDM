import torch
from safetensors.torch import save_file


CKPT_PATH = "./models/epoch=0-step=14253.ckpt"
OUT_PATH = "G:/ComfyUI/models/checkpoints/hdm-xut-340M-1024px-note.safetensors"

print("Loading ckpt...")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
print("Converting...")
state_dict = {
    k: v.half() for k, v in ckpt["state_dict"].items() if not k.startswith("te.")
}
print("Saving...")
save_file(state_dict, OUT_PATH)
