scan_ids=(
    "scene0084_00"
    # "scene0100_00"
    # "scene0164_00"
    # "scene0406_00"
    # "scene0693_00"
)

for scan_id in "${scan_ids[@]}"
do
    echo ">>> process: ${scan_id}"
    
    echo ">>> convert data into a generic format"
    python data_process/scannetv2/export_sensor_data.py --scene_id ${scan_id}
done