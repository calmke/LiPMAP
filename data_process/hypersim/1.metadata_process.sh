
for sid in {001..010}
do
    scene_id=ai_001_${sid}
    echo ">>> process: $scene_id"

    python data_process/hypersim/export_gt_data.py \
        --data_root ./data/Hypersim \
        --save_root ./data/general_data/Hypersim \
        --scene_id ${scene_id}
done