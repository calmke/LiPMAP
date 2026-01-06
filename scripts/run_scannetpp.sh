conf_path=confs/scannetpp.conf
scan_id="0a5c013435"

echo "process scene: ${scan_id}"

python run/train.py \
    --conf $conf_path \
    --scan_id ${scan_id} \
    --exps_folder_name exps_result/scannetpp \
    --expname "${scan_id}" 
