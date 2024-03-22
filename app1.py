import spaces
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

@spaces.GPU
def process(
    pipe,
    path_input,
    ensemble_size,
    denoise_steps,
    processing_res,
    domain,
    normal_out_vis=None,
    path_out_fp32=None,
    path_out_vis=None,
    
):
    if path_out_vis is not None:
        return (
            [normal_out_vis, path_out_vis],
            [normal_out_vis, path_out_fp32, path_out_vis],
        )

    input_image = Image.open(path_input)

    pipe_out = pipe(
        input_image,
        ensemble_size=ensemble_size,
        denoising_steps=denoise_steps,
        processing_res=processing_res,
        domain=domain,
        batch_size=1 if processing_res == 0 else 0,
        show_progress_bar=True,
    )

    depth_pred = pipe_out.depth_np
    depth_colored = pipe_out.depth_colored
    normal_colored = pipe_out.normal_colored
    depth_16bit = (depth_pred * 65535.0).astype(np.uint16)

    path_output_dir = os.path.splitext(path_input)[0] + "_output"
    os.makedirs(path_output_dir, exist_ok=True)

    name_base = os.path.splitext(os.path.basename(path_input))[0]
    path_out_fp32 = os.path.join(path_output_dir, f"{name_base}_depth_fp32.npy")
    normal_out_vis = os.path.join(path_output_dir, f"{name_base}_normal_colored.png")
    path_out_vis = os.path.join(path_output_dir, f"{name_base}_depth_colored.png")

    np.save(path_out_fp32, depth_pred)
    Image.fromarray(depth_16bit).save(path_out_16bit, mode="I;16")
    depth_colored.save(path_out_vis)

    return (
        [normal_out_vis, path_out_vis],
        [normal_out_vis, path_out_fp32, path_out_vis],
    )



def run_demo_server(pipe):
    process_pipe = functools.partial(process, pipe)
    os.environ["GRADIO_ALLOW_FLAGGING"] = "never"

    with gr.Blocks(
        analytics_enabled=False,
        title="Marigold Depth Estimation",
        css="""
            #download {
                height: 118px;
            }
            .slider .inner {
                width: 5px;
                background: #FFF;
            }
            .viewport {
                aspect-ratio: 4/3;
            }
        """,
    ) as demo:
        gr.Markdown(
            """
            <h1 align="center">Marigold Depth Estimation</h1>
            <p align="center">
            <a title="Website" href="https://marigoldmonodepth.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
            </a>
            <a title="arXiv" href="https://arxiv.org/abs/2312.02145" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
            </a>
            <a title="Github" href="https://github.com/prs-eth/marigold" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/github/stars/prs-eth/marigold?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
            </a>
            <a title="Social" href="https://twitter.com/antonobukhov1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
            </a>
            </p>
            <p align="justify">
                Marigold is the new state-of-the-art depth estimator for images in the wild. 
                Upload your image into the <b>left</b> side, or click any of the <b>examples</b> below.
                The result will be computed and appear on the <b>right</b> in the output comparison window.
                <b style="color: red;">NEW</b>: Scroll down to the new 3D printing part of the demo! 
            </p>
        """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image",
                    type="filepath",
                )
                with gr.Accordion("Advanced options", open=False):
                    ensemble_size = gr.Slider(
                        label="Ensemble size",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=1,
                    )
                    denoise_steps = gr.Slider(
                        label="Number of denoising steps",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=10,
                    )
                    processing_res = gr.Radio(
                        [
                            ("Native", 0),
                            ("Recommended", 768),
                        ],
                        label="Processing resolution",
                        value=768,
                    )
                    domain = gr.Radio(
                        [
                            ("indoor", "indoor"),
                            ("outdoor", "outdoor"),
                            ("object", "object"),
                        ],
                        label="scene type",
                        value='indoor',
                    )
                input_output_16bit = gr.File(
                    label="Predicted depth (16-bit)",
                    visible=False,
                )
                input_output_fp32 = gr.File(
                    label="Predicted depth (32-bit)",
                    visible=False,
                )
                input_output_vis = gr.File(
                    label="Predicted depth (red-near, blue-far)",
                    visible=False,
                )
                with gr.Row():
                    submit_btn = gr.Button(value="Compute Depth", variant="primary")
                    clear_btn = gr.Button(value="Clear")
            with gr.Column():
                output_slider = ImageSlider(
                    label="Predicted depth (red-near, blue-far)",
                    type="filepath",
                    show_download_button=True,
                    show_share_button=True,
                    interactive=False,
                    elem_classes="slider",
                    position=0.25,
                )
                files = gr.Files(
                    label="Depth outputs",
                    elem_id="download",
                    interactive=False,
                )

        blocks_settings_depth = [ensemble_size, denoise_steps, processing_res]
        blocks_settings = blocks_settings_depth
        map_id_to_default = {b._id: b.value for b in blocks_settings}

        inputs = [
            input_image,
            ensemble_size,
            denoise_steps,
            processing_res,
            domain,
            input_output_16bit,
            input_output_fp32,
            input_output_vis,

        ]
        outputs = [
            submit_btn,
            input_image,
            output_slider,
            files,
        ]

        def submit_depth_fn(*args):
            out = list(process_pipe(*args))
            out = [gr.Button(interactive=False), gr.Image(interactive=False)] + out
            return out

        submit_btn.click(
            fn=submit_depth_fn,
            inputs=inputs,
            outputs=outputs,
            concurrency_limit=1,
        )


        def clear_fn():
            out = []
            for b in blocks_settings:
                out.append(map_id_to_default[b._id])
            out += [
                gr.Button(interactive=True),
                gr.Button(interactive=True),
                gr.Image(value=None, interactive=True),
                None, None, None, None, None, None, None,
            ]
            return out

        clear_btn.click(
            fn=clear_fn,
            inputs=[],
            outputs=blocks_settings + [
                submit_btn,
                input_image,
                input_output_16bit,
                input_output_fp32,
                input_output_vis,
                output_slider,
                files,
            ],
        )

        demo.queue(
            api_open=False,
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
        )


def main():

    REPO_URL = "https://github.com/lemonaddie/geowizard.git"
    CHECKPOINT = "lemonaddie/Geowizard"
    REPO_DIR = "geowizard"
    
    if os.path.isdir(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    
    repo = git.Repo.clone_from(REPO_URL, REPO_DIR)
    sys.path.append(os.path.join(os.getcwd(), REPO_DIR))

    from pipeline.depth_normal_pipeline_clip_cfg import DepthNormalEstimationPipeline

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    pipe = DepthNormalEstimationPipeline.from_pretrained(CHECKPOINT)
    
    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    pipe = pipe.to('cuda')
    run_demo_server(pipe)


if __name__ == "__main__":
    main()

