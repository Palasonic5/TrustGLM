export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

python train.py --dataset arxiv_semi --model_name graph_llm --llm_frozen False