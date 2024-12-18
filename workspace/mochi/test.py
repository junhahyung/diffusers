import torch
from diffusers import MochiPipeline
from pipeline_mochi_stg import MochiSTGPipeline
from diffusers.utils import export_to_video
import os

# Ensure the samples directory exists
os.makedirs("samples", exist_ok=True)

# Load the pipeline
pipe = STGMochiPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16)

# Enable memory savings
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

# Define parameters
prompt = "A close-up shot of a butterfly landing on the nose of a woman, highlighting her smile and the intricate details of the butterfly's wings, realistic style."
stg_mode = "STG-A"
stg_applied_layers_idx = [35]
stg_scale = 1.0
do_rescaling = True

# Generate video frames
frames = pipe(
    prompt, 
    num_frames=84,
    stg_mode=stg_mode,
    stg_applied_layers_idx=stg_applied_layers_idx,
    stg_scale=stg_scale,
    do_rescaling=do_rescaling
).frames[0]

# Construct the video filename
if stg_scale == 0:
    video_name = f"CFG_rescale_{do_rescaling}.mp4"
else:
    layers_str = "_".join(map(str, stg_applied_layers_idx))
    video_name = f"{stg_mode}_scale_{stg_scale}_layers_{layers_str}_rescale_{do_rescaling}.mp4"

# Save video to samples directory
video_path = os.path.join("samples", video_name)
export_to_video(frames, video_path, fps=30)

print(f"Video saved to {video_path}")
