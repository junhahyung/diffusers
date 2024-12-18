import torch
from diffusers import CogVideoXPipeline
from pipeline_cogvideox_stg import CogVideoXSTGPipeline
from diffusers.utils import export_to_video
import os
import re

# Ensure the samples directory exists
os.makedirs("samples", exist_ok=True)

# Load the pipeline
pipe = CogVideoXSTGPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to("cuda")

# Define common parameters
stg_mode = "STG-A"
stg_scales = [4.0]  # Different scales to iterate
stg_applied_layers_idx_options = [[i] for i in range(30, 31)]  # Different layer indices to iterate
do_rescaling = True
guidance_scale = 6
num_inference_steps = 50

pipe.transformer.to(memory_format=torch.channels_last)

# Specify the path to the text file with prompts
prompt_file_path = "prompts/prompt4.txt"

# Check if the file exists
if not os.path.exists(prompt_file_path):
    raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")

# Read and sanitize prompts from the file
with open(prompt_file_path, "r", encoding="utf-8") as file:
    prompts = [re.sub(r'[^a-zA-Z0-9_\- ]', '_', line.strip()) for line in file if line.strip()]

# Limit to the first 100 prompts
prompts = prompts[:100]

# Process each prompt with different scales and layer indices
for i, prompt in enumerate(prompts):
    print(f"Processing prompt {i+1}/{len(prompts)}: {prompt}")
    
    # Create a subdirectory for the prompt
    sanitized_prompt = prompt[:100]  # Limit folder name length
    prompt_dir = os.path.join("samples", sanitized_prompt)
    os.makedirs(prompt_dir, exist_ok=True)

    for stg_scale in stg_scales:
        for stg_applied_layers_idx in stg_applied_layers_idx_options:
            # Construct the video filename
            layers_str = "_".join(map(str, stg_applied_layers_idx))
            video_name = f"{stg_mode}_scale_{stg_scale}_layers_{layers_str}_rescale_{do_rescaling}.mp4"

            # Save video to the prompt's subdirectory
            video_path = os.path.join(prompt_dir, video_name)
            if os.path.exists(video_path):
                print(f"Video already exists: {video_path}")
                continue
            # Generate video frames
            frames = pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                stg_mode=stg_mode,
                stg_applied_layers_idx=stg_applied_layers_idx,
                stg_scale=stg_scale,
                do_rescaling=do_rescaling,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]

            export_to_video(frames, video_path, fps=8)

            print(f"Video saved to {video_path}")

print("All prompts processed.")
