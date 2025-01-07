import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

torch.backends.cuda.enable_cudnn_sdp(False)

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")

prompt = "Spiderman is surfing"
video_frames = pipe(prompt).frames[0]
video_path = export_to_video(video_frames, "generated.mp4")
video_path