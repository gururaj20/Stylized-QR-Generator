# Paste all this in google colabe file

# 1: Install all required packages
!pip install -q diffusers transformers accelerate torch
!pip install -q xformers
!pip install -q qrcode[pil]
!pip install -q gradio
!pip install -q opencv-python-headless

# 2: Import all necessary libraries
import torch
from PIL import Image, ImageEnhance, ImageFilter
import gradio as gr
import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer, CircleModuleDrawer, GappedSquareModuleDrawer
import numpy as np
from typing import Optional, Tuple

from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler
)

print("Libraries imported successfully!")

# 3: Load AI models
print("Loading AI models... This may take 3-5 minutes.")

controlnet = ControlNetModel.from_pretrained(
    "DionTimmer/controlnet_qrcode-control_v1p_sd15",
    torch_dtype=torch.float16
)

# Text-to-Image pipeline
pipe_txt2img = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)
pipe_txt2img.enable_xformers_memory_efficient_attention()
pipe_txt2img.enable_model_cpu_offload()

# Image-to-Image pipeline
pipe_img2img = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)
pipe_img2img.enable_xformers_memory_efficient_attention()
pipe_img2img.enable_model_cpu_offload()

print("Models loaded successfully!")

# 4: Enhanced QR code generation with multiple styles
def generate_enhanced_qr(
    data: str,
    error_correction: str = "H",
    box_size: int = 10,
    border: int = 4,
    style: str = "squares"
) -> Image.Image:
    """Generate QR code with enhanced error correction and styling"""

    error_levels = {
        "L": qrcode.constants.ERROR_CORRECT_L,  # 7%
        "M": qrcode.constants.ERROR_CORRECT_M,  # 15%
        "Q": qrcode.constants.ERROR_CORRECT_Q,  # 25%
        "H": qrcode.constants.ERROR_CORRECT_H   # 30% (best for artistic QR)
    }

    qr = qrcode.QRCode(
        version=1,
        error_correction=error_levels.get(error_correction, qrcode.constants.ERROR_CORRECT_H),
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # Apply different module styles
    if style == "rounded":
        img = qr.make_image(
            image_factory=StyledPilImage,
            module_drawer=RoundedModuleDrawer()
        )
    elif style == "circles":
        img = qr.make_image(
            image_factory=StyledPilImage,
            module_drawer=CircleModuleDrawer()
        )
    elif style == "gapped":
        img = qr.make_image(
            image_factory=StyledPilImage,
            module_drawer=GappedSquareModuleDrawer()
        )
    else:  # squares (default)
        img = qr.make_image(fill_color="black", back_color="white")

    return img.convert("RGB")

print("QR generation function loaded!")

# 5: Image preprocessing and resizing functions
def preprocess_init_image(image: Image.Image, enhance_contrast: bool = True) -> Image.Image:
    """Enhance initial image for better style transfer"""

    if enhance_contrast:
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)

        # Enhance color
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.2)

        # Slight sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    return image

def resize_for_condition_image(input_image: Image.Image, resolution: int) -> Image.Image:
    """Smart resize maintaining aspect ratio"""
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H = int(round((H * k) / 64.0)) * 64
    W = int(round((W * k) / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

print("Image preprocessing functions loaded!")

# CHUNK 6: Style presets and quality presets
STYLE_PRESETS = {
    "None": ("", ""),
    "Cyberpunk": (
        "cyberpunk style, neon lights, futuristic city, vibrant colors, high contrast, detailed",
        "blurry, low quality, dull colors"
    ),
    "Watercolor": (
        "watercolor painting, soft colors, artistic, flowing, elegant, high quality",
        "harsh lines, digital, pixelated"
    ),
    "Oil Painting": (
        "oil painting, classical art, rich colors, textured, masterpiece, artistic",
        "photograph, digital, flat"
    ),
    "Fantasy": (
        "fantasy art, magical, ethereal, glowing, mystical, detailed, enchanting",
        "realistic, modern, plain"
    ),
    "Nature": (
        "natural landscape, organic, plants, flowers, beautiful scenery, detailed",
        "artificial, geometric, urban"
    ),
    "Abstract": (
        "abstract art, geometric patterns, colorful, modern art, artistic",
        "realistic, photographic"
    ),
    "Steampunk": (
        "steampunk style, brass gears, Victorian era, mechanical, intricate details",
        "modern, digital, simple"
    ),
    "Minimalist": (
        "minimalist design, clean, simple, elegant, modern, high quality",
        "cluttered, complex, busy"
    ),
}

QUALITY_PRESETS = {
    "Fast": {"steps": 30, "guidance": 7.5, "controlnet": 1.3},
    "Balanced": {"steps": 50, "guidance": 10, "controlnet": 1.5},
    "Quality": {"steps": 100, "guidance": 12, "controlnet": 1.7},
    "Maximum": {"steps": 150, "guidance": 15, "controlnet": 2.0},
}

print("Style and quality presets loaded!")
print(f"Available styles: {len(STYLE_PRESETS)-1}")
print(f"Quality levels: {len(QUALITY_PRESETS)}")

# CHUNK 7: Main QR code generation function
def generate_qr_code(
    prompt: str,
    negative_prompt: str,
    qr_data: str,
    init_image: Optional[Image.Image],
    style_preset: str,
    quality_preset: str,
    guidance_scale: float,
    controlnet_scale: float,
    strength: float,
    seed: int,
    error_correction: str,
    qr_style: str,
    enhance_init_image: bool,
    scheduler_type: str
):
    # Validation
    if not qr_data:
        raise gr.Error("QR Code Data cannot be empty.")
    if not prompt and style_preset == "None":
        raise gr.Error("Please enter a prompt or select a style preset.")

    # Apply style preset
    if style_preset != "None":
        style_prompt, style_negative = STYLE_PRESETS[style_preset]
        prompt = f"{prompt}, {style_prompt}" if prompt else style_prompt
        negative_prompt = f"{negative_prompt}, {style_negative}" if negative_prompt else style_negative

    # Apply quality preset
    quality = QUALITY_PRESETS[quality_preset]
    num_steps = quality["steps"]
    if guidance_scale == 10:  # Default value
        guidance_scale = quality["guidance"]
    if controlnet_scale == 1.5:  # Default value
        controlnet_scale = quality["controlnet"]

    print(f"Generating QR code...")
    print(f"Quality: {quality_preset}, Steps: {num_steps}")

    # Generate enhanced QR code
    qr_img = generate_enhanced_qr(qr_data, error_correction, style=qr_style)
    condition_image = resize_for_condition_image(qr_img, 768)

    # Set scheduler
    if scheduler_type == "DDIM":
        scheduler = DDIMScheduler.from_config(pipe_txt2img.scheduler.config)
    elif scheduler_type == "Euler":
        scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_txt2img.scheduler.config)
    else:  # DPM++
        scheduler = DPMSolverMultistepScheduler.from_config(pipe_txt2img.scheduler.config)

    pipe_txt2img.scheduler = scheduler
    pipe_img2img.scheduler = scheduler

    generator = torch.manual_seed(int(seed))

    # Generate image
    if init_image is None:
        print("Using Text-to-Image pipeline")
        image = pipe_txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=condition_image,
            width=768,
            height=768,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=float(controlnet_scale),
            generator=generator,
            num_inference_steps=num_steps,
        ).images[0]
    else:
        print("Using Image-to-Image pipeline")
        if enhance_init_image:
            init_image = preprocess_init_image(init_image)
        init_image_resized = resize_for_condition_image(init_image, 768)

        image = pipe_img2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image_resized,
            control_image=condition_image,
            width=768,
            height=768,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=float(controlnet_scale),
            generator=generator,
            strength=strength,
            num_inference_steps=num_steps,
        ).images[0]

    print("Generation complete!")
    return image

print("Main generation function loaded!")

# 8: Create and launch Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎨 Advanced AI QR Code Generator
    ### Create stunning, scannable QR codes with AI-powered artistic styles
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Basic Settings")

            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe what you want... (e.g., A mystical forest with glowing mushrooms)",
                lines=3
            )

            style_preset = gr.Dropdown(
                choices=list(STYLE_PRESETS.keys()),
                value="None",
                label="Style Preset",
                info="Quick style templates"
            )

            qr_data = gr.Textbox(
                label="QR Code Data",
                placeholder="Enter URL or text (e.g., https://example.com)",
                lines=2
            )

            init_image = gr.Image(
                type="pil",
                label="Initial Style Image (Optional)"
            )

            gr.Markdown("### ⚙ Generation Settings")

            quality_preset = gr.Dropdown(
                choices=list(QUALITY_PRESETS.keys()),
                value="Balanced",
                label="Quality Preset",
                info="Higher quality = slower generation"
            )

            with gr.Accordion("🎛 Advanced Controls", open=False):
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="ugly, disfigured, low quality, blurry, nsfw, pixelated, low contrast",
                    lines=2
                )

                with gr.Row():
                    error_correction = gr.Dropdown(
                        choices=["L", "M", "Q", "H"],
                        value="H",
                        label="Error Correction",
                        info="H = Best for artistic QR"
                    )
                    qr_style = gr.Dropdown(
                        choices=["squares", "rounded", "circles", "gapped"],
                        value="squares",
                        label="QR Pattern Style"
                    )

                guidance_scale = gr.Slider(
                    minimum=1, maximum=20, step=0.5, value=10,
                    label="Guidance Scale",
                    info="How closely to follow the prompt"
                )

                controlnet_scale = gr.Slider(
                    minimum=0.5, maximum=2.5, step=0.1, value=1.5,
                    label="QR Strength",
                    info="Higher = more visible QR pattern"
                )

                strength = gr.Slider(
                    minimum=0.3, maximum=1.0, step=0.05, value=0.8,
                    label="Strength (Image-to-Image)",
                    info="How much to transform the initial image"
                )

                scheduler_type = gr.Dropdown(
                    choices=["DDIM", "Euler", "DPM++"],
                    value="DDIM",
                    label="Scheduler",
                    info="Sampling method"
                )

                seed = gr.Number(
                    label="Seed",
                    value=42,
                    precision=0,
                    info="Use same seed for reproducible results"
                )

                enhance_init_image = gr.Checkbox(
                    label="Enhance Initial Image",
                    value=True,
                    info="Apply preprocessing to initial image"
                )

            submit_button = gr.Button("🚀 Generate QR Code", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 🖼 Generated QR Code")
            output_image = gr.Image(label="Result", type="pil")

            gr.Markdown("""
            ### 💡 Tips for Best Results:
            - Use *High (H)* error correction for artistic QR codes
            - Start with *Balanced* quality preset
            - Increase *QR Strength* if code doesn't scan
            - Try different *Style Presets* for quick results
            - Use *Initial Image* for specific style references
            - Test QR code with multiple scanner apps
            """)

    # Connect button
    submit_button.click(
        fn=generate_qr_code,
        inputs=[
            prompt, negative_prompt, qr_data, init_image,
            style_preset, quality_preset,
            guidance_scale, controlnet_scale, strength, seed,
            error_correction, qr_style, enhance_init_image, scheduler_type
        ],
        outputs=output_image
    )

    gr.Markdown("""
    ---
    ### 📱 How to Use:
    1. Enter your URL/text in *QR Code Data*
    2. Choose a *Style Preset* or write your own prompt
    3. Optionally upload an *Initial Image* for style reference
    4. Click *Generate QR Code*
    5. Test the QR code with your phone!
    """)

# Launch the app
print("🚀 Launching Gradio interface...")
app.launch(share=True, debug=True)
print("✅ App is now running!")
print("🌐 Use the public URL to access from anywhere!")