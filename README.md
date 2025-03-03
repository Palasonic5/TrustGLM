# TrustGLM
Code implementation for paper TrustGLM: Evaluating the Robustness of GraphLLMs Against Prompt, Text, and Structure Attacks

## Environment Preparation

```
conda create --name trustglm python=3.9 -y
conda activate trustglm
conda install peft==0.9

pip install numpy==1.22.1
pip install scipy==1.5.4
pip install boto3 wandb sentencepiece
pip install wandb
```

## Text Attacks
1. Fork the repository https://github.com/RishabhMaheshwary/hard-label-attack and follow its instructions.
2. place the files of textattack in the corresponding repository
3. use the command to do attack

graphprompter:
```
python3 texthoaxer_classification.py --target_model graphprompter --counter_fitting_cos_sim_path mat.txt --USE_cache_path cache_path --max_seq_length 256 --sim_score_window 40 --nclasses 3 --budget 1000 --sampling_portion 0.2 --graphllm_config_file path_to_config_file
```
LLaGA:
```
python3 texthoaxer-llaga.py --target_model llaga --counter_fitting_cos_sim_path mat.txt  --USE_cache_path cache_path --atk_output_dir output_dir --max_seq_length 256 --sim_score_window 40 --nclasses 3 --budget 100000 --sampling_portion 0.2 --graphllm_config_file path_to_config_file
```
Graphtranslator:
```
python3 texthoaxer-translator.py --target_model graphtranslator --counter_fitting_cos_sim_path mat.txt  --USE_cache_path cache_path --atk_output_dir output_dir --max_seq_length 256 --sim_score_window 40 --nclasses 3 --budget 100000 --sampling_portion 0.2 --graphllm_config_file path_to_config_file

```
## Graph Structure Attack
Use the following command to attack surrogate GCN to get perturbed adjacency matrices. (Local method: We generate a distinct perturbed adjacency matrix for each test node. Global method: We generate a single, unified perturbed adjacency matrix shared by all test nodes.)

Nettack:
```
python run_nettack.py --dataset ogbn-arxiv_sup --attr_type sbert --gpu_id 0
```

PRBCD(local):
```
python prbcd_local.py --dataset_name ogbn-arxiv --attr_type sbert
```

PRBCD(global):
```
python prbcd_global.py --dataset_name ogbn-arxiv --attr_type sbert
```
