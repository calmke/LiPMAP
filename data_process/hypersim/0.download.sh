# download hypersim first eight scenes

mkdir -p data/Hypersim
cd data/Hypersim
for sid in {001..010}
do
    scene_id=ai_001_${sid}
    wget https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/${scene_id}.zip
    unzip -q ${scene_id}.zip
    rm ${scene_id}.zip
done
cd ../../
