import gradio as gr
import numpy as np
import os
import sys
import torch
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
from utils_gradio.run_vggt import run_vggt
from utils_gradio.run_detector import run_line_detector
from utils_gradio.run_metric3d import extract_mono_geo_demo
from utils.misc_util import get_class
import shutil
from gradio_rerun import Rerun
import rerun as rr
import rerun.blueprint as rrb
import math
from tqdm import tqdm
from utils.loss_util import normal_loss, metric_depth_loss
from utils_gradio.trainer_util import plot_plane_img
from utils import mesh_util
import random

repo_path = os.path.dirname(__file__)
gradio_tmp_path = os.path.join(repo_path, 'tmp_gradio')
os.environ["GRADIO_TEMP_DIR"] = gradio_tmp_path

if os.path.exists(gradio_tmp_path):
    shutil.rmtree(gradio_tmp_path)
os.makedirs(gradio_tmp_path, exist_ok=False)  # Create the directory if it doesn't exist

def get_recording(recording_id: str) -> rr.RecordingStream:
    return rr.RecordingStream(application_id="rerun_example_gradio", recording_id=recording_id)

def generate_random_string(length=3, characters='abcdef123'):
    return ''.join(random.choice(characters) for _ in range(length))

def opt_one_iter(runner, iter, weight_decay_list, view_info_list, th_d, th_a, log_freq=10):
    # ======================================= process planes
    if iter % runner.process_plane_freq_ite==0:  
        runner.net.regularize_plane_shape()
        runner.net.prune_small_plane()
        if iter > runner.split_start_ite and iter <= runner.max_total_iters - 1000:
            print('splitting...')
            ori_num = runner.net.planarSplat.get_plane_num()
            runner.net.split_plane()
            new_num = runner.net.planarSplat.get_plane_num()
            print(f'plane num: {ori_num} ---> {new_num}')
    # ======================================= get view info
    if not view_info_list:
        view_info_list = runner.dataset.view_info_list.copy()
    view_info = view_info_list.pop(0)

    # ======================================= zero grad
    runner.net.optimizer_zero_grad()
    #  ======================================= calculate losses
    decay = weight_decay_list[iter]
    loss_final, loss_final_dict = runner.net.calculate_loss(view_info, decay, runner.iter_step)
    # loss_final_meters.push(loss_final_dict)
    loss_final.backward()
    runner.net.optimizer_update()
    
    if iter > 0 and iter % runner.check_vis_freq_ite == 0:
        runner.check_plane_visibility_cuda()
    
    # ======================================= plots
    # if iter % runner.plot_freq == 0:
    if iter % log_freq == 0:
        # =================================== plane
        runner.net.regularize_plane_shape()
        runner.net.eval()
        # runner.net.planarSplat.draw_plane(epoch=iter)
        mesh_n, mesh_p = runner.net.planarSplat.draw_plane(epoch=iter)
        runner.plot_plane_img()
        runner.net.train()
        # =================================== line
        # lines3d = runner.net.forward(runner.dataset, 1, 0.01)
        lines3d = runner.net.forward(runner.dataset, th_d, th_a)
        output_path = os.path.join(runner.line_plots_dir, f'final_lines3d_{iter}.ply')
        mesh_util.lines3d2ply(lines3d, output_path)

    else:
        mesh_n, mesh_p, lines3d = None, None, None
    
    # return view_info_list, mesh_n, mesh_p, lines3d
    return view_info_list, mesh_p, lines3d

def run_model(
        image_paths, 
        depth_conf_thresh, 
        iteration_num, 
        init_prim_num, 
        prim_split_thresh, 
        plot_freq, 
        split_freq,
        detector, 
        Dist, 
        Angle, 
        voxel_length, 
        sdf_trunc, 
        depth_trunc,
    ):
    exp_name = generate_random_string()
    rec = get_recording(recording_id=exp_name)
    stream = rec.binary_stream()
    blueprint = rrb.Blueprint(
        # rrb.Horizontal(
        rrb.Vertical(
                # rrb.Spatial3DView(origin="mesh/initial_mesh"),
                rrb.Spatial3DView(origin="mesh/plane"),
                rrb.Spatial3DView(origin="mesh/line"),
                row_shares=[1, 1],
            ),
        # ),
        collapse_panels=True,
    )
    rec.send_blueprint(blueprint)
    # rec.log("mesh/initial_mesh", rr.Clear(recursive=True))
    rec.log("mesh/plane", rr.Clear(recursive=True))
    rec.log("mesh/line", rr.Clear(recursive=True))
    rec.reset_time()
    yield stream.read(), ''
    # rec.log("mesh/initial_mesh", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rec.log("mesh/plane", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rec.log("mesh/line", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rec.set_time("iteration", sequence=0)
    yield stream.read(), ''

    status = "<div style='text-align: center; font-size: 24px;'>Processing Input Images...</div>"
    yield None, status

    out_dir = gradio_tmp_path
    # run vggt
    data = run_vggt(image_paths, depth_conf_thresh=depth_conf_thresh)
    # run metric3dv2
    _, normal_maps_list = extract_mono_geo_demo(data['color'], data['intrinsics'])
    data['normal'] = normal_maps_list
    # 2D line cues
    data = run_line_detector(data, detector)

    # run LiPMAP
    ## load conf
    conf = ConfigFactory.parse_file('confs/vggt_demo.conf')
    conf.put('train.exps_folder_name', out_dir)
    img_res = [data['color'][0].shape[0], data['color'][0].shape[1]]
    conf.put('dataset.scan_id', 'gradio')
    conf.put('dataset.img_res', img_res)
    conf.put('dataset.data', data)
    conf.put('dataset.voxel_length', voxel_length)
    conf.put('dataset.sdf_trunc', sdf_trunc)
    conf.put('dataset.depth_trunc', depth_trunc)
    conf.put('train.max_total_iters', iteration_num)
    conf.put('plane_model.init_plane_num', init_prim_num)
    conf.put('plane_model.split_thres', prim_split_thresh)
    ## run optimization
    exps_folder_name = conf.get_string('train.exps_folder_name')
    runner = get_class(conf.get_string('train.train_runner_class'))(
                                    conf=conf,
                                    batch_size=1,
                                    exps_folder_name=exps_folder_name,
                                    is_continue=False,
                                    timestamp='latest',
                                    do_vis=True,
                                    scan_id='gradio',
                                    resume_path=None,
                                    )
    if runner.start_iter >= runner.max_total_iters:
        return

    runner.process_plane_freq_ite = split_freq

    weight_decay_list = []
    for i in tqdm(range(runner.max_total_iters+1), desc="generating sampling idx list..."):
        weight_decay_list.append(max(math.exp(-i / runner.max_total_iters), 0.1))
    runner.net.train()

    view_info_list = None
    progress_bar = tqdm(range(runner.start_iter, runner.max_total_iters+1), desc="Training progress")
    max_iter = runner.max_total_iters

    for iter in range(runner.start_iter, max_iter + 1):
        runner.iter_step = iter

        if iter == 0:
            runner.check_plane_visibility_cuda()

        # view_info_list, mesh_p, lines3d = opt_one_iter(runner, iter, weight_decay_list, view_info_list, log_freq=plot_freq)
        view_info_list, mesh_p, lines3d = opt_one_iter(runner, iter, weight_decay_list, view_info_list, th_d=Dist, th_a=Angle, log_freq=plot_freq)

        with torch.no_grad():
            # Progress bar
            plane_num = runner.net.planarSplat.get_plane_num()
            if iter % 10 == 0:
                loss_dict = {
                    "Planes": f"{plane_num}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iter == runner.max_total_iters:
                progress_bar.close()

        opt_status = f"<div style='text-align: center; font-size: 24px;'>Optimizing ({iter}/{max_iter})...</div>"
        # if mesh_p is not None and mesh_n is not None:
        if mesh_p is not None:
            print('recording plane mesh...')
            vertex_positions = np.asarray(mesh_p.vertices)
            vertex_colors = np.clip(np.asarray(mesh_p.vertex_colors) * 255, a_min=0, a_max=255).astype(np.uint8)
            triangle_indices = np.asarray(mesh_p.triangles)
            rec.set_time("iteration", sequence=iter)
            rec.log(
                "mesh/plane",
                rr.Mesh3D(
                    vertex_positions=vertex_positions.tolist(),
                    vertex_colors=vertex_colors.tolist(),
                    triangle_indices=triangle_indices.tolist(),
                ),
            )
            yield stream.read(), opt_status
        else:
            yield None, opt_status

        ## line prims
        if lines3d is not None:
            print('recording line mesh...')
            # strips = [lines3d[i].cpu().numpy() for i in range(len(lines3d))]
            strips = []
            for i in range(len(lines3d)):
                seg = lines3d[i].cpu().numpy()
                strips.append(np.array([seg[:3], seg[3:]]))
            rec.set_time("iteration", sequence=iter)
            rec.log(
                "mesh/line",
                rr.LineStrips3D(
                    strips=strips,
                    radii=0.001,
                    colors=[0.9, 0.1, 0.1]
                ),
            )
            yield stream.read(), opt_status
        else:
            yield None, opt_status

def show_image(image_paths, idx=0):
    image_path = image_paths[idx]
    return image_path

def load_test_images(test_images_dir):
    return [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]


TEST_IMAGES_DIR_1 = 'assets/examples_gradio'
img_paths_case1 = load_test_images(TEST_IMAGES_DIR_1)
test_cases = {
    "Test Case 1": {
        "files": img_paths_case1,
    },
}

def load_test_case(test_case_name):
    image_files = test_cases[test_case_name]['files']
    return image_files

css = """
#col-container {
    margin: 0 auto;
    max-width: 800px;
}
"""
_TITLE = "LiPMAP Gradio Demo"
with gr.Blocks(css=css, title=_TITLE) as demo:
    gr.Markdown(f'# {_TITLE}')
    gr.Markdown("3D Reconstruction from Multi-view Images with Line and Plane Primitives")

    with gr.Row():

        with gr.Column():
            gr.Markdown("### Upload Images")
            with gr.Tab(label="Upload Images"):
                test_case_dropdown = gr.Dropdown(
                    label="Select Test Case",
                    choices=list(test_cases.keys()),
                    value='',
                    interactive=True
                )
                with gr.Column():
                    multi_files = gr.File(file_count="multiple", height=100)
            with gr.Row():
                depth_conf_thresh = gr.Slider(label="depth_conf_thresh", value=0.8, minimum=0.1, maximum=20, step=0.1)
                iteration_num = gr.Slider(label="iteration_num", value=3000, minimum=1000, maximum=10000, step=500)
            with gr.Row():
                init_prim_num = gr.Slider(label="init_prim_num", value=2000, minimum=500, maximum=100000, step=500)
                prim_split_thresh = gr.Slider(label="prim_split_thresh", value=0.2, minimum=0.05, maximum=10, step=0.05)
            with gr.Row():
                voxel_length = gr.Number(label="voxel_length", value=0.01)
                sdf_trunc = gr.Number(label="sdf_trunc", value=0.02)
                depth_trunc = gr.Number(label="depth_trunc", value=5.0)
            with gr.Row():
                plot_freq = gr.Slider(label="plot_freq", value=50, minimum=1, maximum=1000, step=1)
                split_freq = gr.Slider(label="split_freq", value=50, minimum=1, maximum=1000, step=1)
                detector = gr.Dropdown(
                                    ['lsd', 'hawpv3', 'deeplsd', 'scalelsd'], 
                                    value='lsd', 
                                    label="Line Detector"
                                )
            with gr.Row():
                Dist = gr.Number(label="Threshold of distance for assignment (Pixels)", value=1)
                Angle = gr.Number(label="Threshold of angle for assignment (Radians)", value=0.01)

        with gr.Column():
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
            )
    test_case_dropdown.change(
        fn=load_test_case,
        inputs=test_case_dropdown,
        outputs=multi_files,
    )

    run_button = gr.Button("Run")
    status_output = gr.Markdown("", label="Running Status")
    run_button.click(
        fn=run_model, 
        inputs=[
            multi_files, 
            depth_conf_thresh, 
            iteration_num, 
            init_prim_num, 
            prim_split_thresh, 
            plot_freq,
            split_freq,
            detector,
            Dist,
            Angle,
            voxel_length,
            sdf_trunc,
            depth_trunc,
        ],
        outputs=[viewer, status_output],
    )

demo.launch()