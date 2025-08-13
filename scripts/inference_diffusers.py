import os
import torch
from hdm.pipeline import HDMXUTPipeline


torch.set_float32_matmul_precision("high")
pipeline = (
    HDMXUTPipeline.from_pretrained(
        "KBlueLeaf/HDM-xut-340M-anime",
        trust_remote_code=True
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
m4 sopmod ii (mod3) (girls' frontline), m4 sopmod ii (girls' frontline), girls' frontline, 
zi ye (hbptcsg2), 

multicolored hair, torn clothes, solo, sand, wide shot, hair between eyes, smile, outdoors, beach, scenery, looking at viewer, dusk, coat, open mouth, cloudy sky, streaked hair, satellite dish, red hair, cityscape, cloud, headgear, sidelocks, ocean, black coat, gloves, floating hair, waves, long hair, wind, torn coat, standing, red eyes, pink hair, sky, city lights, building, skyscraper,

A young girl with long blonde hair, wearing a red and black outfit with horns on her head. the sky is filled with stars and planets, and there are two large satellite dishes on either side of the girl's head. in the background, there is a cityscape with tall buildings and skyscrapers. the overall mood of the image is peaceful and serene.

masterpiece, newest, commentary, photoshop (medium), absurdres
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
        width=1536,
        height=864,
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
