conf_path=confs/hypersim.conf
scan_id="ai_001_001"

echo "process scene: ${scan_id}"

python run/train.py \
    --conf $conf_path \
    --scan_id ${scan_id} \
    --exps_folder_name exps_result/hypersim \
    --expname "${scan_id}" 
