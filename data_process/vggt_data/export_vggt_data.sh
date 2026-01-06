image_path=./data/general_data/general_data/DTU_scan65/images
interval=1 # adjust the number of frames to avoid running OOM

python data_process/vggt_data/run_vggt_demo.py \
    --data_path ${image_path} \
    --frame_step ${interval} \
    --depth_conf 1