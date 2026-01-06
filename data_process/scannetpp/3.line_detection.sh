# [lsd, hawpv3, deeplsd, scalelsd]
detector=scalelsd 
scan_ids=(
    "0a5c013435"
)

for scan_id in "${scan_ids[@]}"
do
    echo ">>> line detection by ${detector}: ${scan_id}"

    # line detection
    python data_process/line_detection.py \
        --data_path ./data/general_data/ScanNetPP/${scan_id}/images \
        --detector ${detector} \
        --save_detected_image
done