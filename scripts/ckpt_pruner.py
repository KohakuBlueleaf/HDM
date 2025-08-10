import torch


CKPT_PATH = "./models/epoch=0-step=14253.ckpt"
OUT_PATH = "./models/hdm-xut-340M-1024px.ckpt"
NO_TE = True
NO_VAE = True

print("Loading ckpt...")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
print("Converting...")
model_sd = ckpt["state_dict"]
for key, val in list(model_sd.items()):
    if NO_TE and key.startswith("te."):
        del model_sd[key]
        continue
    if NO_VAE and key.startswith("vae."):
        del model_sd[key]
        continue
    model_sd[key] = val.half() if val.dtype == torch.float32 else val
state_dict = model_sd
print("Saving...")
torch.save(state_dict, OUT_PATH)
print("Done.")
