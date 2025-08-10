# HDM - Home made Diffusion Models
HDM is a series of models that trained diffusion models (flow matching) from scratch with consumer level hardware in a reasonable cost.
HDM project targeting providing a small but usable base model that can be used for various tasks or perform as a experiment platform or even in practical applications.

## Usage
* requirements: python>=3.10, correct nvidia driver/cuda installed for triton to work.
* Clone this repo
* Install this repo with following option
    * fused: install xformers/liger-kernel for fused operation
    * win: install triton-windows for torch.compile to work
    * tipo: install tipo-kgen and llama.cpp for TIPO prompt gen
* download model file [`hdm-xut-340M-1024px.ckpt`](https://arozos.kblueleaf.net/share/dc426f15-bfdc-4dc4-8cd1-20363816fc24/) to `./models` folder
* start the gradio app

For example:
```bash
git clone https://github.com/KohakuBlueleaf/HDM
cd HDM
python -m venv venv
source venv/bin/activate
# or venv\scripts\activate.ps1 for powershell

# You may want to install pytorch by yourself
# pip install -U torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
# use [..., win] if you are using windows, e.g. [fused,tipo,win]
pip install -e .[fused,tipo]

python ./scripts/inference_fm.py
```

You can use `uv venv` and `uv pip install` as well which will be way more efficient.

## License
This project is still under developement, therefore all the models, source code, text, documents or any media in this project are licensed under `CC-BY-NC-SA 4.0` until the finish of development.