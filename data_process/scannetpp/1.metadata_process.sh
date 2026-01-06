scan_ids=(
    "0a5c013435"
)

for scan_id in "${scan_ids[@]}"
do
    echo ">>> process: ${scan_id}"

    echo ">>> export data from metadata"
    python data_process/scannetpp/prepare_metadata.py \
        --data_root ./data/ScanNet++ \
        --save_root ./data/ScanNetPP/scans \
        --scene_id ${scan_id}
    
    echo ">>> convert data into a generic format"
    python data_process/scannetpp/export_sensor_data.py \
        --data_root ./data/ScanNetPP/scans \
        --save_root ./data/general_data/ScanNetPP \
        --scene_id ${scan_id}
done