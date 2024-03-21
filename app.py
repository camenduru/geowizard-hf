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

from extrude import extrude_depth_3d


def process(
    pipe,
    path_input,
    ensemble_size,
    denoise_steps,
    processing_res,
    path_out_16bit=None,
    path_out_fp32=None,
    path_out_vis=None,
    _input_3d_plane_near=None,
    _input_3d_plane_far=None,
    _input_3d_embossing=None,
    _input_3d_filter_size=None,
    _input_3d_frame_near=None,
):
    if path_out_vis is not None:
        return (
            [path_out_16bit, path_out_vis],
            [path_out_16bit, path_out_fp32, path_out_vis],
        )

    input_image = Image.open(path_input)

    pipe_out = pipe(
        input_image,
        ensemble_size=ensemble_size,
        denoising_steps=denoise_steps,
        processing_res=processing_res,
        batch_size=1 if processing_res == 0 else 0,
        show_progress_bar=True,
    )

    depth_pred = pipe_out.depth_np
    depth_colored = pipe_out.depth_colored
    depth_16bit = (depth_pred * 65535.0).astype(np.uint16)

    path_output_dir = os.path.splitext(path_input)[0] + "_output"
    os.makedirs(path_output_dir, exist_ok=True)

    name_base = os.path.splitext(os.path.basename(path_input))[0]
    path_out_fp32 = os.path.join(path_output_dir, f"{name_base}_depth_fp32.npy")
    path_out_16bit = os.path.join(path_output_dir, f"{name_base}_depth_16bit.png")
    path_out_vis = os.path.join(path_output_dir, f"{name_base}_depth_colored.png")

    np.save(path_out_fp32, depth_pred)
    Image.fromarray(depth_16bit).save(path_out_16bit, mode="I;16")
    depth_colored.save(path_out_vis)

    return (
        [path_out_16bit, path_out_vis],
        [path_out_16bit, path_out_fp32, path_out_vis],
    )


def process_3d(
    input_image,
    files,
    size_longest_px,
    size_longest_cm,
    filter_size,
    plane_near,
    plane_far,
    embossing,
    frame_thickness,
    frame_near,
    frame_far,
):
    if input_image is None or len(files) < 1:
        raise gr.Error("Please upload an image (or use examples) and compute depth first")

    if plane_near >= plane_far:
        raise gr.Error("NEAR plane must have a value smaller than the FAR plane")

    def _process_3d(size_longest_px, filter_size, vertex_colors, scene_lights, output_model_scale=None):
        image_rgb = input_image
        image_depth = files[0]

        image_rgb_basename, image_rgb_ext = os.path.splitext(image_rgb)
        image_depth_basename, image_depth_ext = os.path.splitext(image_depth)

        image_rgb_content = Image.open(image_rgb)
        image_rgb_w, image_rgb_h = image_rgb_content.width, image_rgb_content.height
        image_rgb_d = max(image_rgb_w, image_rgb_h)
        image_new_w = size_longest_px * image_rgb_w // image_rgb_d
        image_new_h = size_longest_px * image_rgb_h // image_rgb_d

        image_rgb_new = image_rgb_basename + f"_{size_longest_px}" + image_rgb_ext
        image_depth_new = image_depth_basename + f"_{size_longest_px}" + image_depth_ext
        image_rgb_content.resize((image_new_w, image_new_h), Image.LANCZOS).save(
            image_rgb_new
        )
        Image.open(image_depth).resize((image_new_w, image_new_h), Image.LANCZOS).save(
            image_depth_new
        )

        path_glb, path_stl = extrude_depth_3d(
            image_rgb_new,
            image_depth_new,
            output_model_scale=size_longest_cm * 10 if output_model_scale is None else output_model_scale,
            filter_size=filter_size,
            coef_near=plane_near,
            coef_far=plane_far,
            emboss=embossing / 100,
            f_thic=frame_thickness / 100,
            f_near=frame_near / 100,
            f_back=frame_far / 100,
            vertex_colors=vertex_colors,
            scene_lights=scene_lights,
        )

        return path_glb, path_stl

    path_viewer_glb, _ = _process_3d(256, filter_size, vertex_colors=False, scene_lights=True, output_model_scale=1)
    path_files_glb, path_files_stl = _process_3d(size_longest_px, filter_size, vertex_colors=True, scene_lights=False)

    # sanitize 3d viewer glb path to keep babylon.js happy
    path_viewer_glb_sanitized = os.path.join(os.path.dirname(path_viewer_glb), "preview.glb")
    if path_viewer_glb_sanitized != path_viewer_glb:
        os.rename(path_viewer_glb, path_viewer_glb_sanitized)
        path_viewer_glb = path_viewer_glb_sanitized

    return path_viewer_glb, [path_files_glb, path_files_stl]


def run_demo_server(pipe):
    process_pipe = functools.partial(process, pipe)
    os.environ["GRADIO_ALLOW_FLAGGING"] = "never"

    with gr.Blocks(
        analytics_enabled=False,
        title="Geowizard Depth and Normal Estimation",
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
                        value=10,
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

        demo_3d_header = gr.Markdown(
            """
            <h3 align="center">Depth Maps</h3>
            <p align="justify">
                TBD 
                result (see Pro Tips below).
            </p>
            """,
            render=False,
        )

        demo_3d = gr.Row(render=False)
        with demo_3d:
            with gr.Column():
                with gr.Accordion("3D printing demo: Main options", open=True):
                    plane_near = gr.Slider(
                        label="Relative position of the near plane (between 0 and 1)",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.001,
                        value=0.0,
                    )
                    plane_far = gr.Slider(
                        label="Relative position of the far plane (between near and 1)",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.001,
                        value=1.0,
                    )
                    embossing = gr.Slider(
                        label="Embossing level",
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=20,
                    )
                with gr.Accordion("3D printing demo: Advanced options", open=False):
                    size_longest_px = gr.Slider(
                        label="Size (px) of the longest side",
                        minimum=256,
                        maximum=1024,
                        step=256,
                        value=512,
                    )
                    size_longest_cm = gr.Slider(
                        label="Size (cm) of the longest side",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=10,
                    )
                    filter_size = gr.Slider(
                        label="Size (px) of the smoothing filter",
                        minimum=1,
                        maximum=5,
                        step=2,
                        value=3,
                    )
                    frame_thickness = gr.Slider(
                        label="Frame thickness",
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=5,
                    )
                    frame_near = gr.Slider(
                        label="Frame's near plane offset",
                        minimum=-100,
                        maximum=100,
                        step=1,
                        value=1,
                    )
                    frame_far = gr.Slider(
                        label="Frame's far plane offset",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=1,
                    )
                with gr.Row():
                    submit_3d = gr.Button(value="Create 3D", variant="primary")
                    clear_3d = gr.Button(value="Clear 3D")
                gr.Markdown(
                    """
                    <h5 align="center">Pro Tips</h5>
                    <ol>
                      <li><b>Re-render with new parameters</b>: Click "Clear 3D" and then "Create 3D".</li>
                      <li><b>Adjust 3D scale and cut-off focus</b>: Set the frame's near plane offset to the 
                          minimum and use 3D preview to evaluate depth scaling. Repeat until the scale is correct and 
                          everything important is in the focus. Set the optimal value for frame's near 
                          plane offset as a last step.</li>
                      <li><b>Increase details</b>: Decrease size of the smoothing filter (also increases noise).</li>
                    </ol>
                    """
                )

            with gr.Column():
                viewer_3d = gr.Model3D(
                    camera_position=(75.0, 90.0, 1.25),
                    elem_classes="viewport",
                    label="3D preview (low-res, relief highlight)",
                    interactive=False,
                )
                files_3d = gr.Files(
                    label="3D model outputs (high-res)",
                    elem_id="download",
                    interactive=False,
                )

        blocks_settings_depth = [ensemble_size, denoise_steps, processing_res]
        blocks_settings_3d = [plane_near, plane_far, embossing, size_longest_px, size_longest_cm, filter_size,
                              frame_thickness, frame_near, frame_far]
        blocks_settings = blocks_settings_depth + blocks_settings_3d
        map_id_to_default = {b._id: b.value for b in blocks_settings}

        inputs = [
            input_image,
            ensemble_size,
            denoise_steps,
            processing_res,
            input_output_16bit,
            input_output_fp32,
            input_output_vis,
            plane_near,
            plane_far,
            embossing,
            filter_size,
            frame_near,
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

        demo_3d_header.render()
        demo_3d.render()

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
                submit_3d,
                input_image,
                input_output_16bit,
                input_output_fp32,
                input_output_vis,
                output_slider,
                files,
                viewer_3d,
                files_3d,
            ],
        )

        def submit_3d_fn(*args):
            out = list(process_3d(*args))
            out = [gr.Button(interactive=False)] + out
            return out

        submit_3d.click(
            fn=submit_3d_fn,
            inputs=[
                input_image,
                files,
                size_longest_px,
                size_longest_cm,
                filter_size,
                plane_near,
                plane_far,
                embossing,
                frame_thickness,
                frame_near,
                frame_far,
            ],
            outputs=[submit_3d, viewer_3d, files_3d],
            concurrency_limit=1,
        )

        def clear_3d_fn():
            return [gr.Button(interactive=True), None, None]

        clear_3d.click(
            fn=clear_3d_fn,
            inputs=[],
            outputs=[submit_3d, viewer_3d, files_3d],
        )

        demo.queue(
            api_open=False,
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
        )




def main():

    if os.path.isdir(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    repo = git.Repo.clone_from(REPO_URL, REPO_DIR)
    repo.git.checkout(REPO_HASH)

    sys.path.append(os.path.join(os.getcwd(), REPO_DIR))

    from diffusers import DiffusionPipeline
    pipeline = DiffusionPipeline.from_pretrained("JUGGHM/temp_repo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = pipeline.from_pretrained(CHECKPOINT)
    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    pipe = pipe.to(device)
    run_demo_server(pipe)


if __name__ == "__main__":
    main()
