import torch
from diffusers import I2VGenXLPipeline
from pipeline_stg_i2vgen_xl import I2VGenXLSTGPipeline
from diffusers.utils import export_to_gif, load_image

torch.backends.cuda.enable_cudnn_sdp(False)
pipeline = I2VGenXLSTGPipeline.from_pretrained(
    "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16"
)
pipeline = pipeline.to("cuda")

image_url = (
    "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
)
image = load_image(image_url).convert("RGB")

prompt = "Papers were floating in the air on a table in the library"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
generator = torch.manual_seed(8888)

frames = pipeline(
    prompt=prompt,
    image=image,
    num_inference_steps=50,
    negative_prompt=negative_prompt,
    guidance_scale=9.0,
    generator=generator,
).frames[0]
video_path = export_to_gif(frames, "i2v.gif")