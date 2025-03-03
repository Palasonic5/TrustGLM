model_path="model/LLaGA/checkpoints/sup-ind/cora_sup/llaga-vicuna-7b-sbert-2-10-2-layer-mlp-projector_nc_epoch3/checkpoint-306"

model_base="model/LLaGA/vicuna-7b-v1.5-16k" #meta-llama/Llama-2-7b-hf
mode="v1" # use 'llaga_llama_2' for llama and "v1" for others
dataset="cora_sup" #test dataset

prompt_file="model/LLaGA/dataset/sup/cora_sup/sampled_2_10_test_prbcd_local.jsonl"
task="nc" #test task
emb="sbert"
use_hop=2 # 2 for ND and 4 for HO
sample_size=10
template="ND"
output_path="model/LLaGA/output/${dataset}/${dataset}_${task}_${template}_prbcd_l.txt"

python Baselines/LLaGA/eval/eval_pretrain.py\
    --model_path ${model_path} \
    --model_base ${model_base} \
    --conv_mode  ${mode} \
    --dataset ${dataset} \
    --pretrained_embedding_type ${emb} \
    --use_hop ${use_hop} \
    --sample_neighbor_size ${sample_size} \
    --answers_file ${output_path} \
    --task ${task} \
    --cache_dir \
    --template ${template}\
    --prompt_file ${prompt_file}
