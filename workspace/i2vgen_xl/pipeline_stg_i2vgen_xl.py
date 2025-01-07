# Copyright 2024 Alibaba DAMO-VILAB and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.models.unets.unet_i2vgen_xl import I2VGenXLUNet
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import (
    BaseOutput,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.i2vgen_xl.pipeline_i2vgen_xl import I2VGenXLPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import I2VGenXLPipeline
        >>> from diffusers.utils import export_to_gif, load_image

        >>> pipeline = I2VGenXLPipeline.from_pretrained(
        ...     "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16"
        ... )
        >>> pipeline.enable_model_cpu_offload()

        >>> image_url = (
        ...     "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
        ... )
        >>> image = load_image(image_url).convert("RGB")

        >>> prompt = "Papers were floating in the air on a table in the library"
        >>> negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
        >>> generator = torch.manual_seed(8888)

        >>> frames = pipeline(
        ...     prompt=prompt,
        ...     image=image,
        ...     num_inference_steps=50,
        ...     negative_prompt=negative_prompt,
        ...     guidance_scale=9.0,
        ...     generator=generator,
        ... ).frames[0]
        >>> video_path = export_to_gif(frames, "i2v.gif")
        ```
"""


@dataclass
class I2VGenXLPipelineOutput(BaseOutput):
    r"""
     Output class for image-to-video pipeline.

    Args:
         frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
             List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
             denoised
     PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
    `(batch_size, num_frames, channels, height, width)`
    """

    frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]]]


class I2VGenXLSTGPipeline(I2VGenXLPipeline):
    def replace_layer_processor(self, down_layers, mid_layers, up_layers, applied_layers_index, replace_processor, initialize=False,):
        """
        Replace the processor for specified layers in the network.

        Args:
            layers (list): A list of tuples where each tuple contains (name, module) of the layer.
            applied_layers_index (list): A list of indices specifying which layers to replace.
            replace_processor (Callable): The processor to replace with.
        """
        if initialize:
            for dl in down_layers:
                dl[1].processor = replace_processor
            for ml in mid_layers:
                ml[1].processor = replace_processor
            for ul in up_layers:
                ul[1].processor = replace_processor
            print(f"[INFO] All layers are replaced with {replace_processor}")
        else:
            for drop_layer in applied_layers_index:
                try:
                    if drop_layer[0] == "d":
                        name, module = down_layers[int(drop_layer[1])]
                    elif drop_layer[0] == "m":
                        name, module = mid_layers[int(drop_layer[1])]
                    elif drop_layer[0] == "u":
                        name, module = up_layers[int(drop_layer[1])]
                    else:
                        raise ValueError(f"Invalid layer type: {drop_layer[0]}")
                    module.processor = replace_processor
                    print(f"[INFO] Layer {name} is replaced with {replace_processor}")

                except IndexError:
                    raise ValueError(
                        f"Invalid layer index: {drop_layer}. "
                    )
            
    def extract_layers(self, temporal=False, file_path="./unet_info.txt"):
        """
        Extract down, mid, and up layers from the UNet model and save to a file.

        Args:
            temporal (bool): Whether to extract temporal layers or not.
            file_path (str): The file path to save layer information.
        
        Returns:
            Tuple: (down_layers, mid_layers, up_layers)
        """
        down_layers, mid_layers, up_layers = [], [], []
        with open(file_path, "w") as f:
            for name, module in self.unet.named_modules():
                if "down" in name or "mid" in name or "up" in name:
                    f.write(f"{name}\n")
                    continue
                if "attn1" in name and "to" not in name:
                    continue
                    if temporal:
                        if "temporal" in name:
                            f.write(f"{name}\n")
                            layer_type = name.split(".")[0].split("_")[0]
                            if layer_type == "down":
                                down_layers.append((name, module))
                            elif layer_type == "mid":
                                mid_layers.append((name, module))
                            elif layer_type == "up":
                                up_layers.append((name, module))
                    else:
                        if "temporal" not in name:
                            f.write(f"{name}\n")
                            layer_type = name.split(".")[0].split("_")[0]
                            if layer_type == "down":
                                down_layers.append((name, module))
                            elif layer_type == "mid":
                                mid_layers.append((name, module))
                            elif layer_type == "up":
                                up_layers.append((name, module))
        return down_layers, mid_layers, up_layers  
    
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = 704,
        width: Optional[int] = 1280,
        target_fps: Optional[int] = 16,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        num_videos_per_prompt: Optional[int] = 1,
        decode_chunk_size: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = 1,
    ):
        r"""
        The call function to the pipeline for image-to-video generation with [`I2VGenXLPipeline`].

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.Tensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            target_fps (`int`, *optional*):
                Frames per second. The rate at which the generated images shall be exported to a video after
                generation. This is also used as a "micro-condition" while generation.
            num_frames (`int`, *optional*):
                The number of video frames to generate.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            eta (`float`, *optional*):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            num_videos_per_prompt (`int`, *optional*):
                The number of images to generate per prompt.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal
                consistency between frames, but also the higher the memory consumption. By default, the decoder will
                decode all frames at once for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, image, height, width, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = guidance_scale
        layers = self.extract_layers(temporal=False)
        assert 0

        # 3.1 Encode input text prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 3.2 Encode image prompt
        # 3.2.1 Image encodings.
        # https://github.com/ali-vilab/i2vgen-xl/blob/2539c9262ff8a2a22fa9daecbfd13f0a2dbc32d0/tools/inferences/inference_i2vgen_entrance.py#L114
        cropped_image = _center_crop_wide(image, (width, width))
        cropped_image = _resize_bilinear(
            cropped_image, (self.feature_extractor.crop_size["width"], self.feature_extractor.crop_size["height"])
        )
        image_embeddings = self._encode_image(cropped_image, device, num_videos_per_prompt)

        # 3.2.2 Image latents.
        resized_image = _center_crop_wide(image, (width, height))
        image = self.video_processor.preprocess(resized_image).to(device=device, dtype=image_embeddings.dtype)
        image_latents = self.prepare_image_latents(
            image,
            device=device,
            num_frames=num_frames,
            num_videos_per_prompt=num_videos_per_prompt,
        )

        # 3.3 Prepare additional conditions for the UNet.
        if self.do_classifier_free_guidance:
            fps_tensor = torch.tensor([target_fps, target_fps]).to(device)
        else:
            fps_tensor = torch.tensor([target_fps]).to(device)
        fps_tensor = fps_tensor.repeat(batch_size * num_videos_per_prompt, 1).ravel()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    fps=fps_tensor,
                    image_latents=image_latents,
                    image_embeddings=image_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                batch_size, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(batch_size, frames, channel, width, height).permute(0, 2, 1, 3, 4)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post processing
        if output_type == "latent":
            video = latents
        else:
            video_tensor = self.decode_latents(latents, decode_chunk_size=decode_chunk_size)
            video = self.video_processor.postprocess_video(video=video_tensor, output_type=output_type)

        # 9. Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return I2VGenXLPipelineOutput(frames=video)


# The following utilities are taken and adapted from
# https://github.com/ali-vilab/i2vgen-xl/blob/main/utils/transforms.py.


def _convert_pt_to_pil(image: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(image, list) and isinstance(image[0], torch.Tensor):
        image = torch.cat(image, 0)

    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image_numpy = VaeImageProcessor.pt_to_numpy(image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_numpy)
        image = image_pil

    return image


def _resize_bilinear(
    image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]], resolution: Tuple[int, int]
):
    # First convert the images to PIL in case they are float tensors (only relevant for tests now).
    image = _convert_pt_to_pil(image)

    if isinstance(image, list):
        image = [u.resize(resolution, PIL.Image.BILINEAR) for u in image]
    else:
        image = image.resize(resolution, PIL.Image.BILINEAR)
    return image


def _center_crop_wide(
    image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]], resolution: Tuple[int, int]
):
    # First convert the images to PIL in case they are float tensors (only relevant for tests now).
    image = _convert_pt_to_pil(image)

    if isinstance(image, list):
        scale = min(image[0].size[0] / resolution[0], image[0].size[1] / resolution[1])
        image = [u.resize((round(u.width // scale), round(u.height // scale)), resample=PIL.Image.BOX) for u in image]

        # center crop
        x1 = (image[0].width - resolution[0]) // 2
        y1 = (image[0].height - resolution[1]) // 2
        image = [u.crop((x1, y1, x1 + resolution[0], y1 + resolution[1])) for u in image]
        return image
    else:
        scale = min(image.size[0] / resolution[0], image.size[1] / resolution[1])
        image = image.resize((round(image.width // scale), round(image.height // scale)), resample=PIL.Image.BOX)
        x1 = (image.width - resolution[0]) // 2
        y1 = (image.height - resolution[1]) // 2
        image = image.crop((x1, y1, x1 + resolution[0], y1 + resolution[1]))
        return image