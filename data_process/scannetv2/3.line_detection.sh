# [lsd, hawpv3, deeplsd, scalelsd]
detector=scalelsd 
scan_ids=(
    "scene0084_00"
    # "scene0100_00"
    # "scene0164_00"
    # "scene0406_00"
    # "scene0693_00"
)

for scan_id in "${scan_ids[@]}"
do
    echo ">>> line detection by ${detector}: ${scan_id}"

    # line detection
    python data_process/line_detection.py \
        --data_path ./data/general_data/ScanNetV2/${scan_id}/images \
        --detector ${detector} \
        --save_detected_image
done