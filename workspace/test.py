from pipeline_mochi_stg import STGMochiPipeline
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
import torch

model_path = "feizhengcong/mochi-1-preview-diffusers"
pipe = MochiPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.to("cuda")
prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
frames = pipe(prompt, 
    num_inference_steps=50, 
    guidance_scale=4.5,
    num_frames=61,
    generator=torch.Generator(device="cuda").manual_seed(42),
    stg_applied_layers_idx=[35],
    stg_scale=2.0,
).frames[0]

export_to_video(frames, "mochi.mp4")
