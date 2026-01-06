conf_path=confs/scannetv2.conf
scan_id="scene0084_00"

echo "process scene: ${scan_id}"

python run/train.py \
    --conf $conf_path \
    --scan_id ${scan_id} \
    --exps_folder_name exps_result/scannetv2 \
    --expname "${scan_id}" 
