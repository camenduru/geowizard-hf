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

REPO_URL = "https://github.com/lemonaddie/geowizard.git"
CHECKPOINT = "lemonaddie/Geowizard"
REPO_DIR = "geowizard"
    
if os.path.isdir(REPO_DIR):
    shutil.rmtree(REPO_DIR)
    
repo = git.Repo.clone_from(REPO_URL, REPO_DIR)
sys.path.append(os.path.join(os.getcwd(), REPO_DIR))

from pipeline.depth_normal_pipeline_clip_cfg import DepthNormalEstimationPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
pipe = DepthNormalEstimationPipeline.from_pretrained(CHECKPOINT)
    
try:
    import xformers
    pipe.enable_xformers_memory_efficient_attention()
except:
    pass  # run without xformers

pipe = pipe.to(device)
#run_demo_server(pipe)

@spaces.GPU
def depth_normal(img,
                denoising_steps,
                ensemble_size,
                processing_res,
                guidance_scale,
                domain):

    #img = img.resize((processing_res, processing_res), Image.Resampling.LANCZOS)
    pipe_out = pipe(
        img,
        denoising_steps=denoising_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        batch_size=0,
        guidance_scale=guidance_scale,
        domain=domain,
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

    _TITLE = '''GeoWizard'''
    _DESCRIPTION = '''
    <div>
    Generate consistent depth and normal from single image.
    <a style="display:inline-block; margin-left: .5em" href='https://github.com/uxiao0719/GeoWizard/'><img src='https://img.shields.io/github/stars/uxiao0719/GeoWizard?style=social' /></a>
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
                    # outputs=[input_image],
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
                         label="Data Domain",
                         value="indoor",
                     )
                        guidance_scale = gr.Slider(
                         label="Classifier Free Guidance Scale",
                         minimum=1,
                         maximum=5,
                         step=1,
                         value=3,
                     )
                        denoising_steps = gr.Slider(
                         label="Number of denoising steps",
                         minimum=1,
                         maximum=20,
                         step=1,
                         value=10,
                     )
                        ensemble_size = gr.Slider(
                         label="Ensemble size",
                         minimum=1,
                         maximum=15,
                         step=1,
                         value=1,
                     )
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
                                guidance_scale,
                                domain],
                        outputs=[depth, normal]
                        )
        demo.queue().launch(share=True, max_threads=80)


if __name__ == '__main__':
    fire.Fire(run_demo)

