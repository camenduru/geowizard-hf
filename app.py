import functools
import os
import shutil
import sys
import git

import gradio as gr
import numpy as np
import torch as torch
from PIL import Image

from gradio_imageslider import ImageSlider

import spaces

import fire

import argparse
import os
import logging

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import glob
import json
import cv2

import sys
sys.path.append("../")
from models.depth_normal_pipeline_clip import DepthNormalEstimationPipeline
from utils.seed_all import seed_all
import matplotlib.pyplot as plt
from utils.de_normalized import align_scale_shift
from utils.depth2normal import *

from diffusers import DiffusionPipeline, DDIMScheduler, AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel

from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# vae = AutoencoderKL.from_pretrained('.', subfolder='vae')
# scheduler = DDIMScheduler.from_pretrained('.', subfolder='scheduler')
# image_encoder = CLIPVisionModelWithProjection.from_pretrained('.', subfolder="image_encoder")
# feature_extractor = CLIPImageProcessor.from_pretrained('.', subfolder="feature_extractor")

stable_diffusion_repo_path = "stabilityai/stable-diffusion-2-1-unclip"
vae = AutoencoderKL.from_pretrained(stable_diffusion_repo_path, subfolder='vae')
scheduler = DDIMScheduler.from_pretrained(stable_diffusion_repo_path, subfolder='scheduler')
sd_image_variations_diffusers_path = 'lambdalabs/sd-image-variations-diffusers'
image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_image_variations_diffusers_path, subfolder="image_encoder")
feature_extractor = CLIPImageProcessor.from_pretrained(sd_image_variations_diffusers_path, subfolder="feature_extractor")
unet = UNet2DConditionModel.from_pretrained('.', subfolder="unet7000")

pipe = DepthNormalEstimationPipeline(vae=vae,
                            image_encoder=image_encoder,
                            feature_extractor=feature_extractor,
                            unet=unet,
                            scheduler=scheduler)

try:
    import xformers
    pipe.enable_xformers_memory_efficient_attention()
except:
    pass  # run without xformers

pipe = pipe.to(device)

@spaces.GPU
def depth_normal(img,
                denoising_steps,
                ensemble_size,
                processing_res,
                seed,
                domain):

    seed = int(seed)
    #torch.manual_seed(seed)

    pipe_out = pipe(
        img,
        denoising_steps=denoising_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        batch_size=0,
        domain=domain,
        seed = seed,
        show_progress_bar=True,
    )

    depth_colored = pipe_out.depth_colored
    normal_colored = pipe_out.normal_colored
    
    return depth_colored, normal_colored



def run_demo():


    custom_theme = gr.themes.Soft(primary_hue="blue").set(
                    button_secondary_background_fill="*neutral_100",
                    button_secondary_background_fill_hover="*neutral_200")
    custom_css = '''#disp_image {
        text-align: center; /* Horizontally center the content */
    }'''

    _TITLE = '''GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image'''
    _DESCRIPTION = '''
    <div>
    Generate consistent depth and normal from single image. High quality and rich details.
    <a style="display:inline-block; margin-left: .5em" href='https://github.com/fuxiao0719/GeoWizard/'><img src='https://img.shields.io/github/stars/fuxiao0719/GeoWizard?style=social' /></a>
    </div>
    '''
    _GPU_ID = 0

    with gr.Blocks(title=_TITLE, theme=custom_theme, css=custom_css) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)
        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                input_image = gr.Image(type='pil', image_mode='RGBA', height=320, label='Input image')

                example_folder = os.path.join(os.path.dirname(__file__), "./files")
                example_fns = [os.path.join(example_folder, example) for example in os.listdir(example_folder)]
                gr.Examples(
                    examples=example_fns,
                    inputs=[input_image],
                    cache_examples=False,
                    label='Examples (click one of the images below to start)',
                    examples_per_page=30
                )
            with gr.Column(scale=1):

                with gr.Accordion('Advanced options', open=True):
                    with gr.Column():
                        
                        domain = gr.Radio(
                         [
                             ("Outdoor", "outdoor"),
                             ("Indoor", "indoor"),
                             ("Object", "object"),
                         ],
                         label="Data Type (Must Select One matches your image)",
                         value="indoor",
                     )
                        denoising_steps = gr.Slider(
                         label="Number of denoising steps (More steps, better quality)",
                         minimum=1,
                         maximum=50,
                         step=1,
                         value=10,
                     )
                        ensemble_size = gr.Slider(
                         label="Ensemble size (1 will be enough. More steps, higher accuracy)",
                         minimum=1,
                         maximum=15,
                         step=1,
                         value=4,
                     )
                        seed = gr.Number(0, label='Random Seed. Negative values for not specifying')
                        
                        processing_res = gr.Radio(
                         [
                             ("Native", 0),
                             ("Recommended", 768),
                         ],
                         label="Processing resolution",
                         value=768,
                     )
                    

                run_btn = gr.Button('Generate', variant='primary', interactive=True)
        with gr.Row():
            with gr.Column():
                depth = gr.Image(interactive=False, show_label=False)
            with gr.Column():
                normal = gr.Image(interactive=False, show_label=False)


        run_btn.click(fn=depth_normal, 
                        inputs=[input_image, denoising_steps,
                                ensemble_size,
                                processing_res,
                                seed,
                                domain],
                        outputs=[depth, normal]
                        )
        demo.queue().launch(share=True, max_threads=80)


if __name__ == '__main__':
    fire.Fire(run_demo)
