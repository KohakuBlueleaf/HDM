from typing import Optional, Tuple, Union

import torch

from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers import AutoencoderKL
from transformers import Qwen3Model, Qwen2Tokenizer

from hdm.modules.xut import XUDiTConditionModel


class HDMXUTPipeline(DiffusionPipeline):
    transformer: XUDiTConditionModel
    tokenizer = Qwen2Tokenizer
    text_encoder: Qwen3Model
    vae: AutoencoderKL

    def __init__(
        self,
        transformer: XUDiTConditionModel,
        text_encoder: Qwen3Model,
        tokenizer: Qwen2Tokenizer,
        vae: AutoencoderKL,
        scheduler,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )
        self.vae_mean = torch.tensor(self.vae.config.latents_mean)[None, :, None, None]
        self.vae_std = torch.tensor(self.vae.config.latents_std)[None, :, None, None]

    def apply_compile(self, *args, **kwargs):
        self.transformer.model.prev_tread_trns = torch.compile(
            self.transformer.model.prev_tread_trns, *args, **kwargs
        )
        self.transformer.model.backbone = torch.compile(
            self.transformer.model.backbone, *args, **kwargs
        )
        self.transformer.model.post_tread_trns = torch.compile(
            self.transformer.model.post_tread_trns, *args, **kwargs
        )
        self.vae.encoder = torch.compile(self.vae.encoder, *args, **kwargs)
        self.vae.decoder = torch.compile(self.vae.decoder, *args, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        prompts: list[str] | str = "a photo of a dog",
        negative_prompts: list[str] | str = "",
        width: int = 1024,
        height: int = 1024,
        cfg_scale: float = 3.0,
        num_inference_steps: int = 16,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.
        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        if len(negative_prompts) == 1:
            negative_prompts = negative_prompts * len(prompts)

        prompt_tokens = self.tokenizer(
            prompts,
            padding="longest",
            return_tensors="pt",
        )
        negative_prompt_tokens = self.tokenizer(
            negative_prompts,
            padding="longest",
            return_tensors="pt",
        )

        prompt_emb = self.text_encoder(
            input_ids=prompt_tokens.input_ids.to(self.device),
            attention_mask=prompt_tokens.attention_mask.to(self.device),
        ).last_hidden_state
        negative_prompt_emb = self.text_encoder(
            input_ids=negative_prompt_tokens.input_ids.to(self.device),
            attention_mask=negative_prompt_tokens.attention_mask.to(self.device),
        ).last_hidden_state

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (
                len(prompts),
                self.transformer.config.input_dim,
                height // 16 * 2,
                width // 16 * 2,
            ),
            generator=generator,
        )
        image = image.to(self.device)
        aspect_ratio = (
            torch.tensor([width / height], device=self.device)
            .log()
            .repeat(image.size(0))
        )

        t = torch.tensor([1] * image.size(0), device=self.device)
        current_t = 1.0
        dt = 1.0 / num_inference_steps

        for _ in (pbar := self.progress_bar(range(num_inference_steps))):
            cond = self.transformer(
                image,
                t,
                prompt_emb,
                added_cond_kwargs={
                    "addon_info": aspect_ratio,
                    "tread_rate": None,
                },
            ).sample
            uncond = self.transformer(
                image,
                t,
                negative_prompt_emb,
                added_cond_kwargs={
                    "addon_info": aspect_ratio,
                    "tread_rate": 0.5,
                },
            ).sample
            cfg_flow = uncond + cfg_scale * (cond - uncond)
            image = image - dt * cfg_flow
            t = t - dt
            current_t -= dt

        torch.cuda.empty_cache()
        image = image * self.vae_std.to(self.device) + self.vae_mean.to(self.device)
        image = torch.concat([self.vae.decode(i[None]).sample for i in image])
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
