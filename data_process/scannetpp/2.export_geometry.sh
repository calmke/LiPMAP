scan_ids=(
    "0a5c013435"
)

for scan_id in "${scan_ids[@]}"
do
    echo ">>> process: ${scan_id}"

    echo ">>> export normal from Omnidata"
    python data_process/export_omnidata_normal.py \
        --images_root ./data/general_data/ScanNetPP/${scan_id}/images

    echo ">>> export normal/depth maps from Metric3D"
    python data_process/export_metric3d_data.py \
        --images_root ./data/general_data/ScanNetPP/${scan_id}/images
    
    echo ">>> export normal/depth maps from MoGe-2"
    python data_process/export_moge2_data.py \
        --images_root ./data/general_data/ScanNetPP/${scan_id}/images
done