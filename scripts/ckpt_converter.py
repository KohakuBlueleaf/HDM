import torch
from safetensors.torch import save_file


CKPT_PATH = "KBlueLeaf/HDM-xut-340M-anime/training_ckpt/768px.ckpt"
OUT_PATH = "KBlueLeaf/HDM-xut-340M-anime/hdm-xut-340M-768px-note.safetensors"

print("Loading ckpt...")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
print("Converting...")
state_dict = {
    k: v.half() for k, v in ckpt["state_dict"].items() if not k.startswith("te.")
}
print("Saving...")
save_file(state_dict, OUT_PATH)
