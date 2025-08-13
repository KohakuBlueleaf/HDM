import os
import random
import omegaconf
import numpy as np

try:
    import triton
except:
    # triton-windows + diffusers/transformers may need
    #   triton pre-import to avoid import error
    print("Triton not found, skip pre import")
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from safetensors import safe_open

torch.set_float32_matmul_precision("medium")
from PIL import Image
from tqdm import trange

import kgen.models as kgen_models
import kgen.executor.tipo as tipo
from kgen.formatter import apply_format, seperate_tags

import xut.env

xut.env.TORCH_COMPILE = True
xut.env.USE_LIGER = True
xut.env.USE_VANILLA = False
xut.env.USE_XFORMERS = True
xut.env.USE_XFORMERS_LAYERS = True
from xut.modules.axial_rope import make_axial_pos

from hdm.loader import load_model
from hdm.trainer import FlowTrainer
from hdm.modules.base import BasicUNet
from hdm.modules.text_encoders import ConcatTextEncoders


DEFAULT_FORMAT = """
<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 
<|quality|>, <|meta|>, <|rating|>,

<|general|>,

<|extended|>.
""".strip()


def prompt_opt(tags, nl_prompt, aspect_ratio, seed):
    meta, operations, general, nl_prompt = tipo.parse_tipo_request(
        seperate_tags(tags.split(",")),
        nl_prompt,
        tag_length_target="long",
        nl_length_target="short",
        generate_extra_nl_prompt=True,
    )
    meta["aspect_ratio"] = f"{aspect_ratio:.3f}"
    result, timing = tipo.tipo_runner(meta, operations, general, nl_prompt, seed=seed)
    return apply_format(result, DEFAULT_FORMAT).strip().strip(".").strip(",")


def cfg_wrapper(
    prompt: str | list[str],
    neg_prompt: str | list[str],
    unet: BasicUNet,
    te: ConcatTextEncoders,
    cfg: float = 5.0,
    zoom: float = 1.0,
    x_shift: float = 0.0,
    y_shift: float = 0.0,
    width: int = 1024,
    height: int = 1024,
):
    _, emb, _, _ = te.encode(prompt, padding="longest")
    _, neg_emb, _, _ = te.encode(neg_prompt, padding="longest")

    batch_size = len(prompt)
    pos_map = make_axial_pos(height, width, device=emb.device).clone()
    pos_map[..., 0] = pos_map[..., 0] + y_shift
    pos_map[..., 1] = pos_map[..., 1] + x_shift
    pos_map = pos_map / zoom
    if pos_map.ndim == 2:
        pos_map = pos_map.unsqueeze(0).expand(batch_size, -1, -1)
    aspect_ratio = (
        torch.tensor([width / height], device=emb.device).log().repeat(batch_size)
    )

    def cfg_fn(x, t):
        cond = unet(
            x,
            t.expand(x.size(0)),
            encoder_hidden_states=emb,
            added_cond_kwargs={
                "addon_info": aspect_ratio,
            },
            pos_map=pos_map,
        )[0].clone()
        uncond = unet(
            x,
            t.expand(x.size(0)),
            encoder_hidden_states=neg_emb,
            added_cond_kwargs={"addon_info": aspect_ratio, "tread_rate": 0.5},
            pos_map=pos_map,
        )[0].clone()
        return uncond + (cond - uncond) * cfg

    return cfg_fn


@torch.no_grad()
def generate(
    nl_prompt: str,
    tag_prompt: str,
    negative_prompt: str,
    tipo_enable: bool,
    format_enable: bool,
    num_images: int,
    steps: int,
    cfg_scale: float,
    size: int,
    aspect_ratio: str,
    fixed_short_edge: bool,
    zoom: float,
    x_shift: float,
    y_shift: float,
    seed: int,
    progress=gr.Progress(),
):
    as_w, as_h = aspect_ratio.split(":")
    aspect_ratio = float(as_w) / float(as_h)
    # Set seed for reproducibility
    if seed is None or seed == -1:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)

    # TIPO
    if tipo_enable:
        tipo.BAN_TAGS = [i.strip() for i in negative_prompt.split(",") if i.strip()]
        final_prompt = prompt_opt(tag_prompt, nl_prompt, aspect_ratio, seed)
    elif format_enable:
        final_prompt = apply_format(nl_prompt, DEFAULT_FORMAT)
    else:
        final_prompt = tag_prompt + "\n" + nl_prompt

    yield None, final_prompt
    all_pil_images = []

    prompts_to_generate = [final_prompt.replace("\n", " ")] * num_images
    negative_prompts_to_generate = [negative_prompt] * num_images

    if fixed_short_edge:
        if aspect_ratio > 1:
            h_factor = 1
            w_factor = aspect_ratio
        else:
            h_factor = 1 / aspect_ratio
            w_factor = 1
    else:
        w_factor = aspect_ratio**0.5
        h_factor = 1 / w_factor

    w = int(size * w_factor / 16) * 2
    h = int(size * h_factor / 16) * 2

    print("=" * 100)
    print(
        f"Generating {num_images} image(s) with seed: {seed} and resolution {w*8}x{h*8}"
    )
    print("-" * 80)
    print(f"Final prompt: {final_prompt}")
    print("-" * 80)
    print(f"Negative prompt: {negative_prompt}")
    print("-" * 80)

    prompts_batch = prompts_to_generate
    neg_prompts_batch = negative_prompts_to_generate

    # Core logic from the original script
    device = trainer_model.device
    unet = trainer_model.unet
    te = trainer_model.te
    vae = trainer_model.vae

    cfg_fn = cfg_wrapper(
        prompts_batch,
        neg_prompts_batch,
        unet=unet,
        te=te,
        cfg=cfg_scale,
        zoom=zoom,
        x_shift=x_shift,
        y_shift=y_shift,
        width=w,
        height=h,
    )
    xt = torch.randn(num_images, 4, h, w).to(device)

    t = 1.0
    dt = 1.0 / steps
    with trange(steps, desc="Generating Steps", smoothing=0.05) as cli_prog_bar:
        for step in progress.tqdm(list(range(steps)), desc="Generating Steps"):
            with torch.autocast(device.type, dtype=torch.float16):
                model_pred = cfg_fn(xt, torch.tensor(t, device=device))
            xt = xt - dt * model_pred.float()
            t -= dt
            cli_prog_bar.update(1)
    torch.cuda.empty_cache()

    generated_latents = xt.float()
    image_tensors = torch.concat(
        [
            vae.decode(
                (
                    generated_latent[None] * trainer_model.vae_std
                    + trainer_model.vae_mean
                ).half()
            ).sample.cpu()
            for generated_latent in generated_latents
        ]
    )
    torch.cuda.empty_cache()

    # Convert tensors to PIL images
    for image_tensor in image_tensors:
        image = Image.fromarray(
            ((image_tensor * 0.5 + 0.5) * 255)
            .clamp(0, 255)
            .numpy()
            .astype(np.uint8)
            .transpose(1, 2, 0)
        )
        all_pil_images.append(image)

    try:
        existing_files = [f for f in os.listdir(output_folder) if f.endswith(".png")]
        if existing_files:
            last = sorted(int(os.path.splitext(f)[0]) for f in existing_files)[-1] + 1
        else:
            last = 0
    except (ValueError, IndexError):
        last = len(os.listdir(output_folder))  # Fallback for non-numeric filenames

    print("-" * 80)
    for idx, image in enumerate(all_pil_images):
        save_path = f"{output_folder}/{idx + last:06}.png"
        image.save(save_path)
        print(f"Saved image to {save_path}")
    print("-" * 80)

    yield all_pil_images, final_prompt


def ui():
    # --- Gradio UI Definition ---
    with gr.Blocks(title="HDM Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# HDM Demo")
        gr.Markdown(
            "### Enter a natural language prompt and/or specific tags to generate an image."
        )
        with gr.Accordion("Introduction", open=False):
            gr.Markdown(
                """
    # HDM: HomeDiffusion Model Project
    HDM is a project to implement a series of generative model that can be pretrained at home.

    ## About this Demo
    This DEMO used a checkpoint during training to demostrate the functionality of HDM.
    Not final model yet.

    ## Usage
    This early model used a model trained on anime image set only, 
    so you should expect to see anime style images only in this demo.

    For prompting, enter danbooru tag prompt to the box "Tag Prompt" with comma seperated and remove the underscore.
    enter natural language prompt to the box "Natural Language Prompt" and enter negative prompt to the box "Negative Prompt".

    If you don't want to spent so much effort on prompting, try to keep "Enable TIPO" selected.

    If you don't want to apply any pre-defined format, unselect "Enable TIPO" and "Enable Format".

    ## Model Spec
    - Backbone: 342M custom DiT(UViT modified) arch
    - Text Encoder: Qwen3 0.6B (596M)
    - VAE: EQ-SDXL-VAE, an EQ-VAE finetuned sdxl vae.

    ## Pretraining Dataset
    - Danbooru 2023 (latest id around 8M)
    - Pixiv famous artist set
    - some pvc figure photos
    """
            )

        with gr.Row():
            with gr.Column(scale=2):
                nl_prompt_box = gr.Textbox(
                    label="Natural Language Prompt",
                    placeholder="e.g., A beautiful anime girl standing in a blooming cherry blossom forest",
                    lines=3,
                )
                tag_prompt_box = gr.Textbox(
                    label="Tag Prompt (comma-separated)",
                    placeholder="e.g., 1girl, solo, long hair, cherry blossoms, school uniform",
                    lines=3,
                )
                neg_prompt_box = gr.Textbox(
                    label="Negative Prompt",
                    value=(
                        "low quality, worst quality, text, signature, jpeg artifacts, bad anatomy, old, early, copyright name, watermark, artist name, signature, weibo username, mosaic censoring, bar censor, censored, text, speech bubbles, realistic"
                    ),
                    lines=3,
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        tipo_enable = gr.Checkbox(
                            label="Enable TIPO",
                            value=True,
                        )
                        format_enable = gr.Checkbox(
                            label="Enable Format",
                            value=True,
                        )
                    with gr.Column(scale=3):
                        with gr.Row():
                            zoom_slider = gr.Slider(
                                label="Zoom",
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.01,
                            )
                            x_shift_slider = gr.Slider(
                                label="X Shift",
                                minimum=-1.0,
                                maximum=1.0,
                                value=0.0,
                                step=0.01,
                            )
                            y_shift_slider = gr.Slider(
                                label="Y Shift",
                                minimum=-1.0,
                                maximum=1.0,
                                value=0.0,
                                step=0.01,
                            )
            with gr.Column(scale=1):
                with gr.Row():
                    num_images_slider = gr.Slider(
                        label="Number of Images", minimum=1, maximum=16, value=1, step=1
                    )
                    steps_slider = gr.Slider(
                        label="Inference Steps", minimum=1, maximum=64, value=32, step=1
                    )

                with gr.Row():
                    cfg_slider = gr.Slider(
                        label="CFG Scale", minimum=1.0, maximum=5.0, value=3.0, step=0.1
                    )
                    seed_input = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        info="Set to -1 for a random seed.",
                    )

                with gr.Row():
                    size_slider = gr.Slider(
                        label="Base Image Size",
                        minimum=768,
                        maximum=1280,
                        value=1024,
                        step=64,
                    )
                with gr.Row():
                    aspect_ratio_box = gr.Textbox(
                        label="Ratio (W:H)",
                        value="1:1",
                    )
                    fixed_short_edge = gr.Checkbox(
                        label="Fixed Edge",
                        value=True,
                    )

                generate_button = gr.Button("Generate", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                output_prompt = gr.TextArea(
                    label="Final Prompt",
                    show_label=True,
                    interactive=False,
                    lines=32,
                    max_lines=32,
                )
            with gr.Column(scale=2):
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=4,
                    rows=3,
                    height="800px",
                )

        generate_button.click(
            fn=generate,
            inputs=[
                nl_prompt_box,
                tag_prompt_box,
                neg_prompt_box,
                tipo_enable,
                format_enable,
                num_images_slider,
                steps_slider,
                cfg_slider,
                size_slider,
                aspect_ratio_box,
                fixed_short_edge,
                zoom_slider,
                x_shift_slider,
                y_shift_slider,
                seed_input,
            ],
            outputs=[output_gallery, output_prompt],
            show_progress_on=output_gallery,
        )
    return demo


def main():
    global trainer_model, device, dtype, output_folder
    model_config_path = "./config/model/xut-qwen3-sm-tread.yaml"
    model_name = "./models/hdm-xut-340M-1024px.ckpt"
    run_name = "xut-small-qwen3-1024px"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    unet, te, tokenizers, vae, scheduler = load_model(
        omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(model_config_path), resolve=True
        )
    )
    trainer_model = (
        FlowTrainer(
            unet=unet,
            te=te,
            vae=vae,
            scheduler=scheduler,
        )
        .eval()
        .requires_grad_(False)
    )

    state_dict = {}
    with safe_open("./models/hdm-xut-340M-1204px-note.safetensors", framework="pt", device=0) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    missing, unexpected = trainer_model.load_state_dict(state_dict, strict=False)
    trainer_model = trainer_model.to(dtype).to(device)

    if unexpected:
        print("Unexpected keys in state dict:", unexpected)

    trainer_model.unet.model.prev_tread_trns = torch.compile(
        trainer_model.unet.model.prev_tread_trns,
        mode="default",
        dynamic=True,
    )
    trainer_model.unet.model.post_tread_trns = torch.compile(
        trainer_model.unet.model.post_tread_trns,
        mode="default",
        dynamic=True,
    )
    trainer_model.unet.model.backbone = torch.compile(
        trainer_model.unet.model.backbone,
        mode="default",
        dynamic=True,
    )
    trainer_model.vae.decoder = torch.compile(
        trainer_model.vae.decoder, mode="default", dynamic=True
    )

    tipo_model_name, gguf_list = kgen_models.tipo_model_list[0]
    kgen_models.download_gguf(tipo_model_name, gguf_list[0])
    kgen_models.load_model(
        f"{tipo_model_name}_{gguf_list[0]}", gguf=True, device="cuda"
    )

    output_folder = f"inference_output/{run_name}"
    os.makedirs(output_folder, exist_ok=True)

    app = ui()
    app.launch()


if __name__ == "__main__":
    main()
