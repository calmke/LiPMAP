# [lsd, hawpv3, deeplsd, scalelsd]
detector=scalelsd 

for sid in {001..010}
do
    scene_id=ai_001_${sid}
    echo ">>> line detection by ${detector}: ${scene_id}"

    python data_process/line_detection.py \
        --data_path ./data/general_data/Hypersim/${scene_id}/images \
        --detector ${detector} \
        --save_detected_image
done