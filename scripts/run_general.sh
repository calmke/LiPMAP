conf_path=confs/general.conf
scan_id="DTU_scan65"

echo "process scene: ${scan_id}"

python run/train.py \
    --conf $conf_path \
    --scan_id ${scan_id} \
    --exps_folder_name exps_result/general \
    --expname "${scan_id}" 
