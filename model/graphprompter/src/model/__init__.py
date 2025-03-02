from graphprompter.src.model.graph_llm import GraphLLM
from graphprompter.src.model.llm import LLM
from graphprompter.src.model.pt_llm import PromptTuningLLM
from graphprompter.src.model.gnn import GCN
from graphprompter.src.model.gnn import GAT
from graphprompter.src.model.llama_adapter import LlamaAdapter
from graphprompter.src.model.t5 import T5

load_model = {
    't5': T5,
    'graph_llm': GraphLLM,
    'llm': LLM,
    'inference_llm': LLM,
    'pt_llm': PromptTuningLLM,
    'gcn': GCN,
    'gat': GAT,
    'llama_adapter': LlamaAdapter,
}


llama_model_path = {
    '7b': '/scratch/jl11523/graphprompter/Llama-2-7b-hf',
    '13b': '[Your LLM PATH]',
    '7b_chat': '[Your LLM PATH]',
    '13b_chat': '[Your LLM PATH]',
}
