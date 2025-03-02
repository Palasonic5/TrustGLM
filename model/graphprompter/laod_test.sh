export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

for dataset in "amazon-sports" "amazon-computers" "amazon-photo"
do
    echo "Processing dataset: $dataset"
    python label_mapping.py --dataset $dataset
done

