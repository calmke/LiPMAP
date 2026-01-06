image_path=$1
echo ">>> process scene ${image_path}"

echo ">>> export normal from Omnidata"
python data_process/export_omnidata_normal.py \
    --images_root image_path

echo ">>> export normal/depth maps from Metric3D"
python data_process/export_metric3d_data.py \
    --images_root image_path

echo ">>> export normal/depth maps from MoGe-2"
python data_process/export_moge2_data.py \
    --images_root image_path

echo ">>> export outputs from VGGT"
interval=1 # adjust the number of frames to avoid running OOM
python data_process/vggt_data/run_vggt_demo.py \
    --data_path ${image_path} \
    --frame_step ${interval} \
    --depth_conf 1               