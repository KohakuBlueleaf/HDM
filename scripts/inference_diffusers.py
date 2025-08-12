import os
import torch
from hdm.pipeline import HDMXUTPipeline


torch.set_float32_matmul_precision("high")
pipeline = (
    HDMXUTPipeline.from_pretrained(
        "KBlueLeaf/HDM-xut-340M-anime",
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
1girl, 
makaino ririmu (3rd costume), makaino ririmu, nijisanji, 
suzue moka, 

very long hair, multicolored hair, long hair, closed mouth, collar, solo, looking at viewer, sky, collarbone, belt collar, demon wings, sidelocks, polka dot, grey hair, red wings, red hair, streaked hair, blush, standing, blue background, bare shoulders, red eyes, strap slip, polka dot background, hair down, dress, sleeveless, blunt bangs, sleeveless dress, straight hair, white dress, pointy ears, flat chest, lace trim, cowboy shot, red collar, sundress, wings, day, fang, smile, bare arms, simple background, lace-trimmed dress, virtual youtuber, hand on own hip, fang out, two-tone hair, ahoge, arm behind head, armpit crease, two side up, arm up,

a vibrant illustration by the artist suzue moka, known for their distinctive style in character design and color usage. The central figure is Makaino Ririmu from the Nijisanji series, depicted with her signature long pink hair adorned with red ribbons. Her pose is confident, with one arm resting behind her head and the other on her hip. The overall composition captures a sense of energy and charm characteristic of suzue moka's artistic style.

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
        width=1024,
        height=2048,
        cfg_scale=3.0,
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
