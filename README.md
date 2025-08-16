# [WIP] HDM - Home made Diffusion Models

[![Source Code - HDM (Here)](https://img.shields.io/badge/Source_Code_(Here)-HDM-2ea44f)](https://github.com/KohakuBlueleaf/HDM)
[![Source Code - HDM(ComfyUI)](https://img.shields.io/badge/Source_Code-HDM(ComfyUI)-2ea44f)](https://github.com/KohakuBlueleaf/HDM-ext)
[![Model - HDM](https://img.shields.io/badge/Model-HDM-2ea44f)](https://huggingface.co/KBlueLeaf/HDM-xut-340M-anime)
[![Document - Tech Report](https://img.shields.io/badge/Document-Tech_Report-2ea44f)](https://github.com/KohakuBlueleaf/HDM/blob/main/TechReport.md)

HDM is a series of models that trained diffusion models (flow matching) from scratch with consumer level hardware in a reasonable cost.
HDM project targeting providing a small but usable base model that can be used for various tasks or perform as a experiment platform or even in practical applications.


![](images/thumbnail.webp)

## Usage

### ComfyUI
* Install this node: https://github.com/KohakuBlueleaf/HDM-ext
* Ensure the transformers library is >= 4.52
    * if you install it from some manager, it should be handled automatically.

### Installation
For local gradio UI or diffusers pipeline inference, you will need to install this repository into your python environment

* requirements: 
    * python>=3.10, python<3.13 (3.13 or higher may not work with pytorch)
    * correct nvidia driver/cuda installed for triton to work.
    * pytorch==2.7.x with triton 3.3.x
    * or, pytorch==2.8.x with triton 3.4.x
    * Optional requirements:
        * TIPO(KGen): llama-cpp-python (may need custom built wheel)
        * liger-kernel: For fused SwiGLU (with torch.compile will works as well)
        * LyCORIS: For lycoris finetune
* Clone this repo
* Install this repo with following option
    * fused: install xformers/liger-kernel for fused operation
    * win: install triton-windows for torch.compile to work
    * tipo: install tipo-kgen and llama.cpp for TIPO prompt gen
* download model file [`hdm-xut-340M-1204px-note.safetensors`](https://huggingface.co/KBlueLeaf/HDM-xut-340M-anime/blob/main/hdm-xut-340M-1024px-note.safetensors) to `./models` folder
* start the gradio app or check the diffusers pipeline inference script
```bash
git clone https://github.com/KohakuBlueleaf/HDM
cd HDM
python -m venv venv
source venv/bin/activate
# or venv\scripts\activate.ps1 for powershell

# You may want to install pytorch by yourself
# pip install -U torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
# use [..., win] if you are using windows, e.g. [fused,tipo,win]
# e.g: pip install -e .[fused,win]
pip install -e .
```
You can use `uv venv` and `uv pip install` as well which will be way more efficient.

### Gradio UI
Once you installed this library with correct dependencies and download the model to `./models` folder.

Run following commands:
```
python ./scripts/inference_fm.py
```

### Diffusers pipeline
hdm library provide a custom pipeline to utilize diffusers' pipeline model format:
```python
import torch
import xut.env

# enable/disable different backend for XUT implementation
# With vanilla/xformers disabled, XUT will use pytorch SDPA attention kernel
xut.env.TORCH_COMPILE = True        # torch.compile for unit module
xut.env.USE_LIGER = False           # Use liger-kernel SwiGLU
xut.env.USE_VANILLA = False         # Use vanilla attention
xut.env.USE_XFORMERS = True         # Use xformers attention
xut.env.USE_XFORMERS_LAYERS = True  # Use xformers SwiGLU

from hdm.pipeline import HDMXUTPipeline

pipeline = (
    HDMXUTPipeline.from_pretrained(
        "KBlueLeaf/HDM-xut-340M-anime", trust_remote_code=True
    )
    .to("cuda:0")
    .to(torch.float16)
)
## Uncomment following line for torch.compile to work on "Whole backbone"
# pipeline.apply_compile(mode="default", dynamic=True)
images = pipeline(
    # Prompts/negative prompts can be list or direct string
    prompts=["1girl, dragon girl, kimono, masterpiece, newest"]*2, 
    negative_prompts="worst quality, low quality, old, early",
    width=1024,
    height=1440,
    cfg_scale=3.0,
    num_inference_steps=24,
    # For camera_param and tread_gamma, check Tech Report for more information.
    camera_param = {
        "zoom": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
    },
    tread_gamma1 = 0.0,
    tread_gamma2 = 0.5,
).images
```

### Training/Finetuning
For both training and finetune you should use `scripts/train.py` script with correct toml config.

For example, you can refer `config/train/hdm-xut-340M-ft.toml` as example lycoris finetune config for HDM-xut-340M 1024px model.

You will need to download the corresponding training_ckpt or safetensors file from HuggingFace repo and fill the file path to model.model_path in the config file.

Then you can run following command:
```
python ./scripts/train.py <train toml config path>
```

**About the dataset**: For simplicity, `hdm.data.kohya.KohyaDataset` support the dataset format which supported by kohya-ss/sd-scripts, while the "repeat" functionality is not implemented yet.

## Next Plan
* UNet-based Hires-Fix/Refiner model
    * new arch specially designed for adaptive resolution text-guided latent refiner
* Use more general dataset (around 40M scale)
    * Currently consider laion-coco-13m + gbc-10m + coyohd-11m + danbooru (total 40M)
    * Will finetune from HDM-xut-340M 256px or 512px ckpt for testing this dataset
* Pretrain a slightly larger model (see tech report, the XUT-large, 555M scale model)

## License
This project is still under developement, therefore all the models, source code, text, documents or any media in this project are licensed under `CC-BY-NC-SA 4.0` until the finish of development.

For any usage that may require any kind of standalone, specialized license. Please directly contact kohaku@kblueleaf.net