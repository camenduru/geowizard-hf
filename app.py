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

@spaces.GPU
def depth_normal(img):

    pipe_out = pipe(
        input_image,
        denoising_steps=10,
        ensemble_size=1,
        processing_res=768,
        batch_size=0,
        guidance_scale=3,
        domain="indoor",
        show_progress_bar=True,
    )

    depth_colored = pipe_out.depth_colored
    normal_colored = pipe_out.normal_colored
    
    return depth_colored, normal_colored

# @spaces.GPU
# def run_demo_server(pipe):
#     title = "Geowizard"
#     description = "Gradio demo for Geowizard."

#     examples = ["files/bee.jpg"]
    
#     # gr.Interface(
#     #     depth_normal, 
#     #     inputs=[gr.Image(type='pil', label="Original Image")], 
#     #     outputs=[gr.Image(type="pil",label="Output Depth"), gr.Image(type="pil",label="Output Normal")], 
#     #     title=title, description=description, article='1', examples=examples, analytics_enabled=False).launch()

    
# def process(
#     pipe,
#     path_input,
#     ensemble_size,
#     denoise_steps,
#     processing_res,
#     path_out_16bit=None,
#     path_out_fp32=None,
#     path_out_vis=None,
# ):

#     if path_out_vis is not None:
#         return (
#             [path_out_16bit, path_out_vis],
#             [path_out_16bit, path_out_fp32, path_out_vis],
#         )

#     input_image = Image.open(path_input)

#     pipe_out = pipe(
#         input_image,
#         denoising_steps=denoise_steps,
#         ensemble_size=ensemble_size,
#         processing_res=processing_res,
#         batch_size=1 if processing_res == 0 else 0,
#         guidance_scale=3,
#         domain="indoor",
#         show_progress_bar=True,
#     )

#     depth_pred = pipe_out.depth_np
#     depth_colored = pipe_out.depth_colored
#     depth_16bit = (depth_pred * 65535.0).astype(np.uint16)

#     path_output_dir = os.path.splitext(path_input)[0] + "_output"
#     os.makedirs(path_output_dir, exist_ok=True)

#     name_base = os.path.splitext(os.path.basename(path_input))[0]
#     path_out_fp32 = os.path.join(path_output_dir, f"{name_base}_depth_fp32.npy")
#     path_out_16bit = os.path.join(path_output_dir, f"{name_base}_depth_16bit.png")
#     path_out_vis = os.path.join(path_output_dir, f"{name_base}_depth_colored.png")

#     np.save(path_out_fp32, depth_pred)
#     Image.fromarray(depth_16bit).save(path_out_16bit, mode="I;16")
#     depth_colored.save(path_out_vis)

#     return (
#         [path_out_16bit, path_out_vis],
#         [path_out_16bit, path_out_fp32, path_out_vis],
#     )


# @spaces.GPU
# def run_demo_server(pipe):
#     process_pipe = functools.partial(process, pipe)
#     os.environ["GRADIO_ALLOW_FLAGGING"] = "never"

#     with gr.Blocks(
#         analytics_enabled=False,
#         title="GeoWizard Depth and Normal Estimation",
#         css="""
#             #download {
#                 height: 118px;
#             }
#             .slider .inner {
#                 width: 5px;
#                 background: #FFF;
#             }
#             .viewport {
#                 aspect-ratio: 4/3;
#             }
#         """,
#     ) as demo:
#         gr.Markdown(
#         """
#             <h1 align="center">Geowizard Depth & Normal Estimation</h1>
#         """
#         )

#         with gr.Row():
#             with gr.Column():
#                 input_image = gr.Image(
#                     label="Input Image",
#                     type="filepath",
#                 )
#                 with gr.Accordion("Advanced options", open=False):
#                     domain = gr.Radio(
#                         [
#                             ("Outdoor", "outdoor"),
#                             ("Indoor", "indoor"),
#                             ("Object", "object"),
#                         ],
#                         label="Data Domain",
#                         value="indoor",
#                     )
#                     cfg_scale = gr.Slider(
#                         label="Classifier Free Guidance Scale",
#                         minimum=1,
#                         maximum=5,
#                         step=1,
#                         value=3,
#                     )
#                     denoise_steps = gr.Slider(
#                         label="Number of denoising steps",
#                         minimum=1,
#                         maximum=20,
#                         step=1,
#                         value=2,
#                     )
#                     ensemble_size = gr.Slider(
#                         label="Ensemble size",
#                         minimum=1,
#                         maximum=15,
#                         step=1,
#                         value=1,
#                     )
#                     processing_res = gr.Radio(
#                         [
#                             ("Native", 0),
#                             ("Recommended", 768),
#                         ],
#                         label="Processing resolution",
#                         value=768,
#                     )
#                 input_output_16bit = gr.File(
#                     label="Predicted depth (16-bit)",
#                     visible=False,
#                 )
#                 input_output_fp32 = gr.File(
#                     label="Predicted depth (32-bit)",
#                     visible=False,
#                 )
#                 input_output_vis = gr.File(
#                     label="Predicted depth (red-near, blue-far)",
#                     visible=False,
#                 )
#                 with gr.Row():
#                     submit_btn = gr.Button(value="Compute", variant="primary")
#                     clear_btn = gr.Button(value="Clear")
#             with gr.Column():
#                 output_slider = ImageSlider(
#                     label="Predicted depth (red-near, blue-far)",
#                     type="filepath",
#                     show_download_button=True,
#                     show_share_button=True,
#                     interactive=False,
#                     elem_classes="slider",
#                     position=0.25,
#                 )
#                 files = gr.Files(
#                     label="Depth outputs",
#                     elem_id="download",
#                     interactive=False,
#                 )

#         blocks_settings_depth = [ensemble_size, denoise_steps, processing_res]
#         blocks_settings = blocks_settings_depth
#         map_id_to_default = {b._id: b.value for b in blocks_settings}

#         inputs = [
#             input_image,
#             ensemble_size,
#             denoise_steps,
#             processing_res,
#             input_output_16bit,
#             input_output_fp32,
#             input_output_vis,
#         ]
#         outputs = [
#             submit_btn,
#             input_image,
#             output_slider,
#             files,
#         ]

#         def submit_depth_fn(*args):
#             out = list(process_pipe(*args))
#             out = [gr.Button(interactive=False), gr.Image(interactive=False)] + out
#             return out

#         submit_btn.click(
#             fn=submit_depth_fn,
#             inputs=inputs,
#             outputs=outputs,
#             concurrency_limit=1,
#         )

#         gr.Examples(
#             fn=submit_depth_fn,
#             examples=[
#                 [
#                     "files/bee.jpg",
#                     10,  # ensemble_size
#                     10,  # denoise_steps
#                     768,  # processing_res
#                     "files/bee_depth_16bit.png",
#                     "files/bee_depth_fp32.npy",
#                     "files/bee_depth_colored.png",
#                 ],
#             ],
#             inputs=inputs,
#             outputs=outputs,
#             cache_examples=True,
#         )

#         def clear_fn():
#             out = []
#             for b in blocks_settings:
#                 out.append(map_id_to_default[b._id])
#             out += [
#                 gr.Button(interactive=True),
#                 gr.Image(value=None, interactive=True),
#                 None, None, None, None, None, None, None,
#             ]
#             return out

#         clear_btn.click(
#             fn=clear_fn,
#             inputs=[],
#             outputs=blocks_settings + [
#                 submit_btn,
#                 input_image,
#                 input_output_16bit,
#                 input_output_fp32,
#                 input_output_vis,
#                 output_slider,
#                 files,
#             ],
#         )

#         demo.queue(
#             api_open=False,
#         ).launch(
#             server_name="0.0.0.0",
#             server_port=7860,
#         )


def main():

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

    title = "Geowizard"
    description = "Gradio demo for Geowizard."
    examples = ["files/bee.jpg"]

    gr.Interface(
        depth_normal, 
        inputs=[gr.Image(type='pil', label="Original Image")], 
        outputs=[gr.Image(type="pil",label="Output Depth"), gr.Image(type="pil",label="Output Normal")], 
        title=title, description=description, article='1', examples=examples, analytics_enabled=False).launch()


if __name__ == "__main__":
    main()
