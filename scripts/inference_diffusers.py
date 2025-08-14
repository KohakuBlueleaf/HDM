import os
import torch
from hdm.pipeline import HDMXUTPipeline


torch.set_float32_matmul_precision("high")
pipeline = (
    HDMXUTPipeline.from_pretrained(
        "KBlueLeaf/HDM-xut-340M-anime", trust_remote_code=True
    )
    .to("cuda:0")
    .to(torch.float16)
)
# uncomment following line for torch.compile to work
# pipeline.apply_compile(mode="default", dynamic=True)


with torch.autocast("cuda", torch.float16):
    image = pipeline(
        [
            """
1girl, nagato (azur lane), azur lane, dan-98,
animal, outdoors, black hair, animal ear fluff, tree, wide sleeves, nature, umbrella, yellow eyes, night, long hair, forest, fur-trimmed kimono, snow, oil-paper umbrella, snowing, fur trim, winter, solo, oversized animal, blunt bangs, fox ears, hime cut, wooden bridge, looking at viewer, scenery, red kimono, kimono, fox, long sleeves, fox girl, animal ears, japanese clothes, straight hair, bridge, bare tree,

The image, created by the artist dan-98, depicts a serene winter scene from the Azur Lane series. It features two characters standing on a wooden bridge that spans over a snowy forest. One character is holding an oil-paper umbrella with a red and white pattern, providing shelter to both figures. The backdrop of bare trees and falling snowflakes enhances the tranquil atmosphere of this picturesque winter landscape.

masterpiece, newest, absurdres
""".strip().replace(
                "\n", " "
            )
        ]
        * 4,
        """
low quality, worst quality, text, signature, jpeg artifacts, bad anatomy, old, early, 
copyright name, watermark, artist name, signature, weibo username, mosaic censoring, 
bar censor, censored, text, speech bubbles, hair intake, 
realistic, 2girls, 3girls, multiple girls, crop top, cropped head, cropped
""".strip().replace(
            "\n", " "
        ),
        width=1280,
        height=960,
        cfg_scale=3.0,
        tread_gamma1=0.25,
        tread_gamma2=0.75,
        camera_param={
            "zoom": 0.95,
            "x_shift": 0.0,
            "y_shift": -0.1,
        },
        num_inference_steps=32,
    )


output_folder = "./inference_output/diffusers-pipeline"
try:
    existing_files = [f for f in os.listdir(output_folder) if f.endswith(".png")]
    if existing_files:
        last = sorted(int(os.path.splitext(f)[0]) for f in existing_files)[-1] + 1
    else:
        last = 0
except (ValueError, IndexError):
    last = len(os.listdir(output_folder))  # Fallback for non-numeric filenames

print("-" * 80)
for idx, image in enumerate(image.images):
    save_path = f"{output_folder}/{idx + last:06}.png"
    image.save(save_path)
    print(f"Saved image to {save_path}")
print("-" * 80)
