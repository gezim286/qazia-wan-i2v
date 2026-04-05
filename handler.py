import runpod
import torch
import base64
import os
import tempfile
import subprocess
from pathlib import Path

MODEL_ID_720P  = "Wan-AI/Wan2.1-I2V-14B-720P"
MODEL_ID_480P  = "Wan-AI/Wan2.1-I2V-14B-480P"
CACHE_DIR      = "/runpod-volume/models"

pipeline = None
loaded_resolution = None

def load_model(resolution="720p"):
    global pipeline, loaded_resolution
    if pipeline is not None and loaded_resolution == resolution:
        return pipeline
    from wan.pipelines import WanI2VPipeline
    model_id = MODEL_ID_720P if resolution in ("720p", "1080p") else MODEL_ID_480P
    print(f"Loading Wan I2V {model_id}...")
    pipeline = WanI2VPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    ).to("cuda")
    loaded_resolution = resolution
    print("Wan I2V loaded.")
    return pipeline

def b64_to_image(b64_string):
    from PIL import Image
    from io import BytesIO
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(BytesIO(img_bytes)).convert("RGB")

def handler(job):
    job_input = job.get("input", {})

    # Quick health-check mode — no model download needed
    if job_input.get("test_mode"):
        return {"status": "ok", "message": "handler ready"}

    prompt          = job_input.get("prompt", "cinematic motion")
    negative_prompt = job_input.get("negative_prompt", "blurry, static")
    resolution      = job_input.get("resolution", "720p")
    duration        = int(job_input.get("duration", 5))
    first_frame_b64 = job_input.get("first_frame_b64", "")
    first_frame_url = job_input.get("first_frame_url", "")

    # Load the first frame image
    if first_frame_b64:
        image = b64_to_image(first_frame_b64)
    elif first_frame_url:
        import requests
        from PIL import Image
        from io import BytesIO
        resp = requests.get(first_frame_url, timeout=30)
        resp.raise_for_status()
        from io import BytesIO
        image = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        return {"error": "first_frame_b64 or first_frame_url is required"}

    res_map = {"480p": (832, 480), "720p": (1280, 720), "1080p": (1280, 720)}
    width, height = res_map.get(resolution, (1280, 720))
    image = image.resize((width, height))

    p = load_model(resolution)
    frames = p(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=duration * 16,
        width=width,
        height=height,
        guidance_scale=5.0,
        num_inference_steps=20,
    ).frames[0]

    # Save as MP4
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "output.mp4")
        from diffusers.utils import export_to_video
        export_to_video(frames, out_path, fps=16)
        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"video_b64": video_b64, "format": "mp4"}

runpod.serverless.start({"handler": handler})
