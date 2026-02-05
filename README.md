# LiP-Map: Interacted Planes Reveal 3D Line Mapping
<div align="center">

<a href="https://arxiv.org/abs/2602.01296"><img src="https://img.shields.io/badge/ArXiv-2602.01296-brightgreen"></a>&ensp;<a href="https://calmke.github.io/LiPMAP/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a>


[Zeran Ke](https://calmke.github.io/)<sup>1,2*</sup>, [Bin Tan](https://icetttb.github.io/)<sup>2*</sup>, [Gui-Song Xia](http://www.captain-whu.com/zh/person/xiaguisong.html)<sup>1</sup>,  [Yujun Shen](https://shenyujun.github.io/)<sup>2</sup>, [Nan Xue](https://xuenan.net/)<sup>2†</sup>

<sup>1</sup>Wuhan University &ensp;&ensp;<sup>2</sup>Ant Group

</div>

<!-- <img src="assets/teaser.jpg" width="100%"> -->

![teaser](assets/teaser.png)

## 📖 Overview
3D line mapping from multi-view RGB images provides a compact and structured visual representation of scenes. We study the problem from a physical and topological perspective: **a 3D line most naturally emerges as the edge of a finite 3D planar patch.** We present LiP-Map, a line–plane joint optimization framework that explicitly models learnable line and planar primitives. This coupling enables accurate and detailed 3D line mapping while maintaining strong efficiency (typically completing a reconstruction in 3 to 5 minutes per scene). LiP-Map pioneers the integration of planar topology into 3D line mapping, not by imposing pairwise coplanarity constraints but by explicitly constructing interactions between plane and line primitives, thus offering a principled route toward structured reconstruction in man-made environments.

## ⚙️ Installtion

### 1.Clone repo
```
git clone https://github.com/ant-research/LiPMAP.git --recursive 
```

### 2.Create the enviroment
```
conda create -n lipmap python=3.10
conda activate lipmap

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt 
pip install submodules/diff-rect-rasterization
pip install submodules/quaternion-utils

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install "git+https://github.com/cherubicXN/hawp.git"
pip install "git+https://github.com/cherubicXN/easylsd.git"
pip install "git+https://github.com/microsoft/MoGe.git"
pip install "git+https://github.com/facebookresearch/vggt.git"
```


## 📊 Data Preparation
We provide all scripts for pre-processing on ScanNetV2, ScanNet++, Hypersim, and your own multi-view images.
Please download and place all datasets in the `./data` folder.

### 1. ScanNetV2

1.Please follow instructions reported in [SimpleRecon](https://github.com/nianticlabs/simplerecon) to download and extract ScanNetV2 scenes. The extacted data format should be like:
```
data/ScanNetV2
    scans/
        scene0084_00/
            scene0000_00_vh_clean_2.ply (gt mesh)
            sensor_data/
                frame-000000.color.jpg 
                frame-000000.depth.jpg 
                frame-000000.pose.txt
            scene0000.txt (scan metadata and image sizes)
            intrinsic/
                intrinsic_depth.txt
                intrinsic_color.txt
        scene0000_01/
        ....
```

2.Run scripts as follow:
```
bash data_process/scannetv2/1.metadata_process.sh
bash data_process/scannetv2/2.export_geometry.sh
bash data_process/scannetv2/3.line_detection.sh
```

### 2. ScanNet++

1.Please follow instructions reported in [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) to download and extract ScanNet++ scenes. The extacted data format should be like:
```
data/ScanNet++
    45b0dac5e3/
        iphone/
        scans/
    16c9bd2e1e/
    ....
```

2.Run scripts as follow:
```
bash data_process/scannetpp/1.metadata_process.sh
bash data_process/scannetpp/2.export_geometry.sh
bash data_process/scannetpp/3.line_detection.sh
```

### 3. Hypersim

1.Please download Hypersim use the following scrpit:
```
bash data_process/hypersim/0.download.sh
```
The extracted data format should be like:
```
data/Hypersim
      ai_001_001/
            images/
                  scene_cam_00_final_hdf5/
                  scene_cam_00_final_preview/
                  scene_cam_00_geometry_hdf5/
                  scene_cam_00_geometry_preview/
      ai_001_002/
      ....
      ai_001_010/
      ...
```

2.Run scripts as follow:
```
bash data_process/hypersim/1.metadata_process.sh
bash data_process/hypersim/2.line_detection.sh
```


- After the above preprocessing, data from different sources are stored in **the same unified format**. The final data format should be like:
```
data/general_data/{dataset_name}/
      {scene_id}/
            images/
                  000000.png
                  ....
            {depth_type}_depth/
                  000000_depth.png
                  000000_depth.npy
                  ....
            {normal_type}_normal/
                  000000_normal.png
                  000000_normal.npy
                  ....
            poses/
                  000000.txt
                  ....
            {line_detector}/
                  000000.json
                  000000.png
                  ....
            intrinsics.txt
```
- `dataset_name: [ScanNetV2, ScanNetPP, Hypersim]`

- `depth_type: [gt/sensor, metric3d_large, moge2]`

- `normal_type: [metric3d_large, moge2, omnidata]`

- `line_detector: [lsd, hawpv3, deeplsd, scalelsd]`

### 4. General data with only images

We use VGGT to obtain camera intrinsics, poses, normal/depth maps. Set the `image_path` in the script and run the script:
```
bash data_process/vggt_data/export_vggt_data.sh 
```
The final data format should be like:
```
data/general_data/general_data/
      images/
            000000.png
            ....
      vggt/
            images/
                  000000.png
                  ....
            vggt_depth/
                  000000_depth.png
                  000000_depth.npy
                  ....
            vggt_normal/
                  000000_normal.png
                  000000_normal.npy
                  ....
            poses/
                  000000.txt
                  ....
            {line_detector}/
                  000000.json
                  000000.png
                  ....
            intrinsics.txt
```

## 🚗🔥Reconstruction
Set the ***scene_id*** in the scripts and run scripts as follow:

### ScanNetV2
```
bash scripts/run_scannetv2.sh
```

### ScanNet++
```
bash scripts/run_scannetpp.sh
```

### Hypersim
```
bash scripts/run_hypersim.sh
```
### General data
```
bash scripts/run_general.sh
```


## 🔍 Visualization 
We provide standard and convenient API used to visualize the variables involved in the process of our 3d line mapping in the file `./utils/mesh_util.py`, and the common used one is:
```
mesh_util.vis_plane_rays_juncs_lines([OPTICS])
OPTICS:
      plane_corners: the vertex of the planes, used to draw the planar primitives. [N_p, 4, 3]
      rays: [rays_o, rays_o], used to draw the rays. rays_d: [N_r, 3], rays_o: [N_r, 3]
      juncs: used to draw the selected junctions. [N_j, 3]
      lines, lines2: used to together or separately draw two sets of lines. [N_l, 4]
      t: length of rays, which is used to control how long the rays drawn in open3d.
      use_color: if True, use color for visualization
      view_info_list: list of input view information, used to draw the cameras.
```
Set the `workdir` and run the following script to obtain the final 3D lines:
```
bash scripts/vis_plane_lines.sh
```
It should be **noted** that set `dist_th=5` yields better visualizations when VGGT produces poor outputs.

## 💻 Quick Gradio Demo
Run the following command to open the Gradio interface in the local browser:
```
python run_gradio_demo.py
```
Defaultly select `Test Case 1`, set `Line Detector`, and set thresholds of `distance` and `angle`.
Then click the `Run` botton.

<img src="assets/figs/scalelsd-testcase1.png" alt="Gradio Demo Interface" width="800">

Also, you can upload images and set optimization parameters for final 3D reconstruction.
Here, we take 49 images of the scene `scan16` from `DTU dataset` as an example.
<img src="assets/figs/deeplsd-dtu-scan16.png" alt="Gradio Demo Interface" width="800">

## 🙏 Acknowledgements
LiPMAP is built on the top of several outstanding open-source projects. We are extremely grateful for the contributions of these projects and their communities, whose hard work has greatly propelled the development of the field and enabled our work to be realized.
- [PlanarSplatting](https://github.com/ant-research/PlanarSplatting)
- [Omnidata](https://github.com/EPFL-VILAB/omnidata)
- [Metric3D v2](https://github.com/YvanYin/Metric3D)
- [VGGT](https://github.com/facebookresearch/vggt)
- [MoGe](https://github.com/microsoft/MoGe)
- [HAWP](https://github.com/cherubicXN/hawp)
- [DeepLSD](https://github.com/cvg/DeepLSD)
- [ScaleLSD](https://github.com/ant-research/scalelsd)
