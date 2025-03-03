from src.model.graph_llm import GraphLLM
from src.model.graph_llm_adv import GraphLLMADV
from src.model.graph_llm_pgd import GraphLLMPGD
from src.model.llm import LLM
from src.model.pt_llm import PromptTuningLLM
from src.model.gnn import GCN
from src.model.gnn import GAT
from src.model.llama_adapter import LlamaAdapter
from src.model.t5 import T5

load_model = {
    't5': T5,
    'graph_llm': GraphLLM,
    'graph_llm_adv': GraphLLMADV,
    'graph_llm_pgd': GraphLLMPGD,
    'llm': LLM,
    'inference_llm': LLM,
    'pt_llm': PromptTuningLLM,
    'gcn': GCN,
    'gat': GAT,
    'llama_adapter': LlamaAdapter,
}


llama_model_path = {
    '7b': '/scratch/xs2334/TrustGLM/Baselines/graphprompter/Llama-2-7b-hf',
    '13b': '[Your LLM PATH]',
    '7b_chat': '[Your LLM PATH]',
    '13b_chat': '[Your LLM PATH]',
}
