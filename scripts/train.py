import warnings
import sys
import random
from typing import *

import numpy as np

try:
    import triton
except:
    print("Triton not found, skip pre import")
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torch._dynamo.config
import lightning.pytorch as pl
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    GradientAccumulationScheduler,
)
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from safetensors import safe_open

from hdm.modules.text_encoders import ConcatTextEncoders
from hdm.utils import instantiate
from hdm.utils.config import load_train_config
from hdm.loader import load_model
from hdm.trainer import FlowTrainer
from hdm.trainer.callbacks import ImageGenCallback
from hdm.data.base import CombineDataset

from lycoris import create_lycoris


torch._dynamo.config.recompile_limit = 1024
torch._dynamo.config.accumulated_recompile_limit = 8192
torch.set_float32_matmul_precision("medium")
warnings.filterwarnings("ignore", ".*sequence length is longer than.*")


def cfg_wrapper(
    prompt: str | list[str],
    neg_prompt: str | list[str],
    width: int,
    height: int,
    unet,
    te: ConcatTextEncoders,
    cfg: float = 5.0,
    use_normed_emb: bool = False,
):
    with torch.autocast("cuda", dtype=next(te.parameters()).dtype):
        normed_emb, emb, pool, mask = te.encode(
            prompt, padding="max_length", truncation=True
        )
        neg_normed_emb, neg_emb, neg_pool, neg_mask = te.encode(
            neg_prompt, padding="max_length", truncation=True
        )
    if use_normed_emb:
        emb = normed_emb
        neg_emb = neg_normed_emb

    if pool is not None:
        # For SDXL-like arch
        added_cond = {
            "time_ids": torch.tensor([[height, width, 0, 0, height, width]])
            .repeat(2, 1)
            .to(emb),
            "text_embeds": torch.concat([pool, neg_pool]),
        }
    else:
        added_cond = {"addon_info": torch.log(torch.tensor([width / height])).to(emb)}

    if emb.size(1) > neg_emb.size(1):
        pad_setting = (0, 0, 0, emb.size(1) - neg_emb.size(1))
        neg_emb = F.pad(neg_emb, pad_setting)
        if neg_mask is not None:
            neg_mask = F.pad(neg_mask, pad_setting[2:])
    if neg_emb.size(1) > emb.size(1):
        pad_setting = (0, 0, 0, neg_emb.size(1) - emb.size(1))
        emb = F.pad(emb, pad_setting)
        if mask is not None:
            mask = F.pad(mask, pad_setting[2:])

    if mask is not None and neg_mask is not None:
        attn_mask = torch.concat([mask, neg_mask])
    else:
        attn_mask = None
    text_ctx_emb = torch.concat([emb, neg_emb])

    def cfg_fn(x, t):
        cond, uncond = unet(
            torch.concat([x, x]),
            t.expand(x.size(0) * 2),
            encoder_hidden_states=text_ctx_emb,
            encoder_attention_mask=attn_mask,
            added_cond_kwargs=added_cond,
        ).chunk(2)
        return uncond + (cond - uncond) * cfg

    return cfg_fn


@torch.inference_mode()
def sampling(pl_module: FlowTrainer, batch, config):
    te = pl_module.te
    bs = config["batch_size"]
    captions = batch[1][: config["num"]]
    if len(captions) < config["num"]:
        captions = (captions * (config["num"] // len(captions) + 1))[: config["num"]]

    images = []
    device = pl_module.device
    vae.to(device)
    size = config.get("size", 1024)

    with torch.autocast("cuda"):
        for i in range(0, config["num"], bs):
            current_caption = captions[i : i + bs]
            cfg_fn = cfg_wrapper(
                current_caption,
                [""] * len(current_caption),
                width=size,
                height=size,
                unet=lambda *args, **kwargs: pl_module.unet(*args, **kwargs)[0],
                te=te,
                cfg=2.5,
                use_normed_emb=pl_module.te_use_normed_ctx,
            )
            xt = torch.randn(len(current_caption), 4, size // 8, size // 8).to(device)
            t = 1.0
            dt = 1.0 / config["steps"]
            for _ in range(config["steps"]):
                model_pred = cfg_fn(xt, torch.tensor(t, device=device))
                xt = xt - dt * model_pred
                t -= dt
            generated_latents = xt - LATENT_SHIFT
            image_tensors = torch.concat(
                [
                    vae.decode(
                        latent[None] * pl_module.vae_std + pl_module.vae_mean
                    ).sample
                    * 0.5
                    + 0.5
                    for latent in generated_latents
                ]
            ).float()
            # convert to PIL
            for image_tensor in image_tensors:
                image = Image.fromarray(
                    (image_tensor * 255)
                    .cpu()
                    .clamp(0, 255)
                    .numpy()
                    .astype(np.uint8)
                    .transpose(1, 2, 0)
                )
                images.append(image)
    return captions, images


def main(config_path):
    global LATENT_SHIFT, SCALING_FACTOR, vae
    previous_trainer_model = None
    model, dataset, trainer, lightning = load_train_config(config_path)
    LATENT_SHIFT = model["latent_shift"]
    SCALING_FACTOR = model["scaling_factor"]
    EPOCH = lightning["epochs"]
    GPUS = lightning["devices"]
    GRAD_ACC = lightning["grad_acc"]
    inf_dtype = instantiate(model["inference_dtype"])
    max_grad_acc = GRAD_ACC if isinstance(GRAD_ACC, int) else max(GRAD_ACC.values())

    pl.seed_everything(lightning.get("seed", random.randint(0, 2**32 - 1)))

    unet, te, tokenizers, vae, scheduler = load_model(model["config"])
    vae.config.scaling_factor = model["scaling_factor"]
    vae.eval().to(inf_dtype).cpu()
    te.eval().to(inf_dtype).cpu()
    if "lycoris" in model:
        lycoris = create_lycoris(unet, **model["lycoris"])
        lycoris.apply_to()
        unet.to(inf_dtype).cpu().requires_grad_(False)
    else:
        lycoris = None

    ds = CombineDataset(
        [instantiate(dataset) for dataset in dataset["datasets"]],
        latent_scale=dataset["scaling_factor"],
        latent_shift=dataset["latent_shift"],
        tokenizers=tokenizers,
        shuffle=True,
        **dataset.get("base_kwargs", {}),
    )
    loader = Data.DataLoader(
        ds,
        batch_size=lightning["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=lightning["dataloader_workers"],
        persistent_workers=(
            lightning.get("persistent_workers", False)
            and bool(lightning["dataloader_workers"])
        ),
        pin_memory=True,
        collate_fn=ds.collate,
        prefetch_factor=max_grad_acc if bool(lightning["dataloader_workers"]) else None,
    )

    training_batch_per_epoch = len(loader) // (
        GPUS if isinstance(GPUS, int) else len(GPUS)
    )
    print("Batches per epoch: ", training_batch_per_epoch)
    if isinstance(GRAD_ACC, int):
        training_step = training_batch_per_epoch // GRAD_ACC * EPOCH + EPOCH
        grad_acc = {0: GRAD_ACC}
    elif isinstance(GRAD_ACC, dict):
        grad_acc = {}
        training_step = 0
        current_epoch = 0
        current_acc = 1
        for i in range(EPOCH):
            if i in GRAD_ACC:
                current_acc = GRAD_ACC[i]
            grad_acc[i] = current_acc
            training_step += training_batch_per_epoch // current_acc + 1
    print(GRAD_ACC, grad_acc)

    if (
        "end" in trainer.get("lr_sch_configs", {})
        and trainer["lr_sch_configs"]["end"] < 0
    ):
        trainer["lr_sch_configs"]["end"] = training_step + 1
    if "lr" not in trainer["lr_sch_configs"]:
        trainer["lr_sch_configs"] = {"lr": dict(**trainer["lr_sch_configs"])}

    print("Total training step: ", training_step)

    logger = None
    logger = WandbLogger(**lightning["logger"])
    if "ckpt_path" in model:
        trainer_model = FlowTrainer.load_from_checkpoint(
            model["ckpt_path"],
            unet=unet,
            te=te,
            vae=vae,
            scheduler=scheduler,
            lycoris_model=lycoris,
            **trainer,
            full_config={
                "model": model,
                "dataset": dataset,
                "lightning": lightning,
                "trainer": trainer,
            },
            strict=False,
            map_location="cpu",
        ).cpu()
    else:
        trainer_model = FlowTrainer(
            unet=unet,
            te=te,
            vae=vae,
            scheduler=scheduler,
            lycoris_model=lycoris,
            **trainer,
            full_config={
                "model": model,
                "dataset": dataset,
                "lightning": lightning,
                "trainer": trainer,
            },
        ).cpu()
        if "model_path" in model:
            path = model["model_path"]
            if path.endswith(".safetensors"):
                with safe_open(path, framework="pt", device="cpu") as f:
                    state_dict = {k: f.get_tensor(k) for k in f.keys()}
            else:
                state_dict = torch.load(path, map_location="cpu")
            missing, unexpected = trainer_model.load_state_dict(
                state_dict, strict=False
            )
            if unexpected:
                print("Unexpected keys: ", unexpected)

    vae = trainer_model.vae
    te = trainer_model.te
    vae.eval().to(inf_dtype).cpu().requires_grad_(False)
    te.eval().to(inf_dtype).cpu().requires_grad_(False)
    if lycoris is not None:
        unet = trainer_model.unet
        unet.eval().to(inf_dtype).cpu().requires_grad_(False)
    torch.cuda.empty_cache()

    if getattr(trainer_model, "lycoris_model", None) is not None:
        logger.watch(trainer_model.lycoris_model, log="all", log_freq=32)
    else:
        logger.watch(trainer_model.unet, log="all", log_freq=32)

    if previous_trainer_model is not None:
        trainer_model.unet.load_state_dict(previous_trainer_model.unet.state_dict())
    if lightning.get("grad_ckpt", False):
        trainer_model.unet.enable_gradient_checkpointing()

    steps = lightning.get("max_steps", None)
    if (GPUS if isinstance(GPUS, int) else len(GPUS)) > 1:
        if isinstance(GPUS, int):
            GPUS = list(range(GPUS))
        strategy = DDPStrategy(
            parallel_devices=[torch.device(f"cuda:{i}") for i in GPUS],
            gradient_as_bucket_view=True,
            ddp_comm_hook=default.fp16_compress_hook,
        )
    else:
        strategy = "auto"
    trainer = pl.Trainer(
        max_epochs=None if steps is not None else EPOCH,
        max_steps=steps or -1,
        accelerator="gpu",
        devices=GPUS,
        precision=lightning["precision"],
        gradient_clip_val=lightning["grad_clip"],
        logger=logger,
        log_every_n_steps=1,
        strategy=strategy,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ImageGenCallback(lightning.get("imggencallback", {}), sampling),
            ModelCheckpoint(every_n_train_steps=500),
            ModelCheckpoint(every_n_epochs=1),
            GradientAccumulationScheduler(grad_acc),
        ],
    )
    trainer.fit(
        trainer_model,
        loader,
        ckpt_path=lightning.get("ckpt_path", None),
    )


if __name__ == "__main__":
    main(sys.argv[1])
