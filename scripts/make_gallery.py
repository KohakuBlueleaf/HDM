import os
import sys
sys.path.append(".")
import shutil
import json
import random
from PIL import Image, ExifTags
from template.content_template import (
    selectors,
    images,
    thumbnail_inp,
    thumbnail_nav,
    slides,
)


def get_exif(image):
    exif_info = image.getexif()
    result = {}
    for tag, value in exif_info.items():
        result[ExifTags.TAGS.get(tag, "Unknown "+str(tag))] = value
    for tag, value in exif_info.get_ifd(34665).items():
        if isinstance(value, bytes):
            value = value.strip(b"UNICODE").replace(b"\x00", b"").decode()
        result[ExifTags.TAGS.get(tag, "Unknown "+str(tag))] = value
    return result


def get_png_info(image):
    info = image.text
    if "workflow" in info:
        workflow = json.loads(info["workflow"])
        nodes = workflow["nodes"]
        for node in nodes:
            if "ShowText|pysssss" == node["type"]:
                return node["widgets_values"][0][0]
    if "prompt" in info:
        prompt = json.loads(info["prompt"])
        prompts = []
        find_target = False
        for k, node in prompt.items():
            if "CLIPTextEncode" == node['class_type']:
                text = node['inputs']['text']
                if isinstance(text, list):
                    target_id = text[0]
                    target_widget_id = text[1]
                    find_target = True
                else:
                    prompts.append(text)
        if not find_target and len(prompts)>1:
            if "low quality" in prompts[0]:
                return prompts[-1]
            else:
                return prompts[0]
        else:
            for k, node in prompt.items():
                if k == target_id:
                    if "text" in node["inputs"]:
                        return node["inputs"]["text"]
                    text = node.get('widgets_values', [""])[target_widget_id]
    return ""


PATH = "images/samples"
MODEL_SAMPLE_PATH = "./KBlueLeaf/HDM-xut-340M-anime/images/samples"


with open("./template/gallery_template.md", "r", encoding="utf-8") as f:
    base_template = f.read()

all_selectors = []
all_images = []
all_thumbnail_inp = []
all_thumbnail_nav = []
all_slides = []


thumbnail = Image.open("./images/thumbnail.png")
thumbnail.save("./images/thumbnail.webp", quality=100)
thumbnail.save("./KBlueLeaf/HDM-xut-340M-anime/images/thumbnail.webp", quality=100)

index = 1
shutil.rmtree(MODEL_SAMPLE_PATH, ignore_errors=True)
os.makedirs(MODEL_SAMPLE_PATH, exist_ok=True)
imgs = os.listdir(PATH)
random.seed(32)
random.shuffle(imgs)
for img in imgs:
    if not any(img.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
        continue

    image_path = f"{PATH}/{img}"
    image = Image.open(image_path)
    exif = get_exif(image)

    title = ""
    description = str(exif.get("UserComment", "")).replace("\n", "<br><br>\n")
    if not description:
        description = get_png_info(image)

    if not description:
        continue

    shutil.copyfile(os.path.join(PATH, img), os.path.join(MODEL_SAMPLE_PATH, img))

    description = f"Prompt: {description}"

    all_selectors.append(selectors.format(index=index))
    all_images.append(images.format(index=index, path=image_path))
    all_thumbnail_inp.append(
        thumbnail_inp.format(index=index, status="checked" if index == 1 else "")
    )
    all_thumbnail_nav.append(thumbnail_nav.format(index=index))
    all_slides.append(
        slides.format(
            index=index, path=image_path, title=title, description=description
        )
    )

    index += 1
print(index)

result = (
    base_template
    .replace("{selectors}", ",\n".join(all_selectors))
    .replace("{images}", "\n".join(all_images))
    .replace("{thumbnail_inp}", "\n".join(all_thumbnail_inp))
    .replace("{thumbnail_nav}", "\n".join(all_thumbnail_nav))
    .replace("{slides}", "\n".join(all_slides))
)

with open("./KBlueLeaf/HDM-xut-340M-anime/README_no_gallery.md", "r", encoding="utf-8") as f:
    base_template = f.read()

with open("./KBlueLeaf/HDM-xut-340M-anime/README.md", "w", encoding="utf-8") as f:
    result = (
        base_template
        .replace("{gallery}", result)
        .replace(PATH, f"https://huggingface.co/KBlueLeaf/HDM-xut-340M-anime/resolve/main/{PATH}")
    )
    f.write(result)