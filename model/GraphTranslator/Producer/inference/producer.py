import csv
import argparse
import time
import logging
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import random
from torch_geometric.utils import degree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.log_utils import setup_logging
from utils.env import init_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, required=True)
    parser.add_argument('--llm_checkpoint', type=str, default="Baselines/GraphTranslator/Translator/models/chatglm-6b", required=False)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--distributed", action='store_const', default=False, const=True)
    parser.add_argument('--random_seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_workers", default=1, type=int)

    return parser.parse_args()


args = parse_args()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def calculate_degrees(edge_index, num_nodes):
    """
    计算每个节点的度。
    
    Args:
        edge_index (torch.Tensor): 边索引张量，形状为 [2, num_edges]。
        num_nodes (int): 图中节点的数量。
    
    Returns:
        torch.Tensor: 每个节点的度，形状为 [num_nodes]。
    """
    # 计算节点的度
    deg = degree(edge_index[0], num_nodes=num_nodes)
    return deg

def clean_text(text):
    """
    清理文本中的特殊符号，使其能被 LLM 正确处理。
    
    Args:
        text (str): 原始文本。
    
    Returns:
        str: 清理后的文本。
    """
    symbols_to_remove = ['≤', '≥', '≠', '∫', '∞', '√']
    for symbol in symbols_to_remove:
        text = text.replace(symbol, '')
    return text

# 加载数据
def read_dataset(dataset_name):
    data = torch.load(f"Baselines/GraphTranslator/data/{dataset_name}/processed_data.pt")

    raw_texts = [clean_text(text) for text in data.raw_texts]
    titles = [clean_text(title) for title in data.title]

    titles = ["Title:" if title.strip() == "" else title for title in titles]

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    deg = calculate_degrees(edge_index, num_nodes)
    
    data_df = pd.DataFrame({
        "node_id": np.arange(num_nodes),  # 节点 ID
        "title": titles,
        "title_abstract": raw_texts
    })

    edge_index = data.edge_index.numpy()
    src_nodes, dst_nodes = edge_index[0], edge_index[1]
    neighbor_df = pd.DataFrame({"src_node": src_nodes, "dst_node": dst_nodes})
    sampled_neighbors = sample_neighbors(neighbor_df, deg, k=5)

    return data_df, sampled_neighbors

def sample_neighbors(neighbor_df, deg, k=5):
    """
    对邻居进行加权采样。
    
    Args:
        neighbor_df (pd.DataFrame): 包含 src_node 和 dst_node 的 DataFrame。
        deg (np.ndarray): 节点的度数组，索引对应节点 ID。
        k (int): 每个源节点采样的目标节点数量。
    
    Returns:
        pd.DataFrame: 包含采样后的邻居关系。
    """
    sample_neighbor_list = []

    grouped_neighbors = neighbor_df.groupby('src_node')

    for src, group in grouped_neighbors:
        dst_list = group['dst_node'].to_list()  
        deg_list = deg[dst_list].numpy()  
        deg_sum = np.sum(deg_list) 

        if len(dst_list) <= k:
            sampled_dst = dst_list
        else:
            deg_prob = deg_list / deg_sum
            sampled_dst = random.choices(dst_list, weights=deg_prob, k=k)

        sampled_group = group[group['dst_node'].isin(sampled_dst)]
        sample_neighbor_list.append(sampled_group)

    sampled_neighbors = pd.concat(sample_neighbor_list, ignore_index=True)
    return sampled_neighbors

def read_arxiv_dataset():
    # paperid to node的映射和node to paperid的映射
    node2paperid = {}
    paperid2node = {}
    with open('../../data/arxiv/arxiv_nodeidx2paperid.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            nodeIdx = int(row[0])
            paperId = int(row[1])
            node2paperid[nodeIdx] = paperId
            paperid2node[paperId] = nodeIdx

    # 读取paperId到title和abstract映射的内容
    paperId2titleAndabs = pd.read_csv("../../data/arxiv/titleabs.tsv", delimiter='\t', header=None)
    paperId2titleAndabs = paperId2titleAndabs.rename(columns={0: "paper_id", 1: "title", 2: "abstract"})
    paperId2titleAndabs['node_id'] = paperId2titleAndabs['paper_id'].map(paperid2node).fillna(-1).astype(int)
    paperId2titleAndabs["title_abstract"] = "Title: " + paperId2titleAndabs["title"] + "\n" +"Abstract: " + paperId2titleAndabs["abstract"]
    paperId2titleAndabs = paperId2titleAndabs[paperId2titleAndabs['node_id'] != -1]

    paperId2titleAndabs = paperId2titleAndabs.replace('≤', '', regex=True)
    paperId2titleAndabs = paperId2titleAndabs.replace('≥', '', regex=True)
    paperId2titleAndabs = paperId2titleAndabs.replace('≠', '', regex=True)
    paperId2titleAndabs = paperId2titleAndabs.replace('≠', '', regex=True)
    paperId2titleAndabs = paperId2titleAndabs.replace('∫', '', regex=True)
    paperId2titleAndabs = paperId2titleAndabs.replace('∞', '', regex=True)
    paperId2titleAndabs = paperId2titleAndabs.replace('√', '', regex=True)

    sorted_paperId2titleAndabs = paperId2titleAndabs.sort_values(by='node_id')
    sample_neighbor_df = pd.read_csv("../../data/arxiv/sample_neighbor_df.csv")

    return sorted_paperId2titleAndabs, sample_neighbor_df


class LLM(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self._args = args
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self._args.llm_checkpoint, trust_remote_code=True)
        # model
        self.llm = AutoModel.from_pretrained(self._args.llm_checkpoint, trust_remote_code=True).half().to(device)

    def inference_chatglm(self, input_data, sample_neighbor_df, dataset_name, train_ids):
        self.llm.eval()

        node_title_and_abs = input_data.set_index('node_id')['title_abstract'].to_dict()
        src_to_dst_dict = sample_neighbor_df.groupby('src_node')['dst_node'].apply(list).to_dict()
        node2title = input_data.set_index('node_id')['title'].to_dict()

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} total paper count: {input_data.shape[0]}")
        summary = []
        for data in input_data.iterrows():
            node_id = data[1]['node_id']
            if node_id in train_ids:
                title = data[1]['title']
                src_prompt_pre = "The title and abstract of this paper are as follows: "
                src_prompt = '\n please summarize this paper and list five key words of this paper. All answers are in English and No Chinese in your answer'
                src_title_abstract = data[1]['title_abstract']
                node_word_input = src_prompt_pre + src_title_abstract
                if len(node_word_input[0]) > 3000- len(src_prompt):
                    node_word_input = node_word_input[:3000-len(src_prompt)]
                node_word_input += src_prompt

                dst_prompt_pre = '\n The paper title and abstract are provided as follows: '
                dst_prompt = "\n Please summarize the topic and content of these papers. All answers are in English and No Chinese in your answer"
                dst_title_abstract = ""
                for neighbor_id in src_to_dst_dict[node_id]:
                    dst_title_abstract = dst_title_abstract + node_title_and_abs[neighbor_id] + '\n'

                neighbor_word_input  = dst_prompt_pre + dst_title_abstract
                if len(neighbor_word_input[0]) > 3000-len(dst_prompt):
                    neighbor_word_input = neighbor_word_input[:3000-len(dst_prompt)]
                neighbor_word_input += dst_prompt

                try:
                    response_node, _ = self.llm.chat(self.tokenizer,
                                                            node_word_input ,
                                                            history=[])
                    response_neighbor, _ = self.llm.chat(self.tokenizer,
                                                                neighbor_word_input,
                                                                history=[])
                    summary.append({
                        'node_id': node_id,
                        'title': title,
                        'response_node': response_node,
                        'response_neighbor': response_neighbor
                    })
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} paper {node_id+1} title: \"{title}\"")
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("CUDA out of memory error detected, skipping this batch")
                        continue
                    else:
                        continue

        summary_df = pd.DataFrame(summary)
        embeddings = torch.load(f"Baselines/GraphTranslator/data/{dataset_name}/graphsage_node_embeddings.pt").to('cpu')
        new_data = []
        for _, row in summary_df.iterrows():
            node_id = int(row['node_id'])
            embedding = np.array(embeddings[node_id].detach())
            str_array = [str(num) for num in embedding]
            str_representation = ", ".join(str_array)
            title = node2title[row['node_id']]

            new_data.append({
                'node_id': node_id,
                'embedding':str_representation ,
                'paper_summary':row['response_node'],
                'citepapers_summary':row['response_neighbor'],
                'title':title
                })
        summary_embeddings = pd.DataFrame(new_data)
        summary_embeddings.to_csv(f'Baselines/GraphTranslator/data/{dataset_name}/train_summary_embeddings.csv',index=False)

# 加载数据
def read_dataset_products(dataset_name):
    data = torch.load(f"Baselines/GraphTranslator/data/{dataset_name}/processed_data.pt")

    raw_texts = [clean_text(text) for text in data.raw_texts]

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    deg = calculate_degrees(edge_index, num_nodes)
    
    # 创建 DataFrame，包含节点 ID、标题和摘要
    data_df = pd.DataFrame({
        "node_id": np.arange(num_nodes),  # 节点 ID
        # "title": titles,
        "pro_desc": raw_texts
    })

    # 创建邻居关系 DataFrame
    edge_index = data.edge_index.numpy()
    src_nodes, dst_nodes = edge_index[0], edge_index[1]
    neighbor_df = pd.DataFrame({"src_node": src_nodes, "dst_node": dst_nodes})
    sampled_neighbors = sample_neighbors(neighbor_df, deg, k=5)

    return data_df, sampled_neighbors


class proLLM(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self._args = args
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self._args.llm_checkpoint, trust_remote_code=True)
        # model
        self.llm = AutoModel.from_pretrained(self._args.llm_checkpoint, trust_remote_code=True).half().to(device)

    def inference_chatglm(self, input_data, sample_neighbor_df, dataset_name, train_ids):
        self.llm.eval()

        node_name_and_desc = input_data.set_index('node_id')['pro_desc'].to_dict()
        src_to_dst_dict = sample_neighbor_df.groupby('src_node')['dst_node'].apply(list).to_dict()
        # node2title = input_data.set_index('node_id')['title'].to_dict()

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} total paper count: {input_data.shape[0]}")
        summary = []
        for data in input_data.iterrows():
            node_id = data[1]['node_id']
            if node_id in train_ids:
                # title = data[1]['title']
                # product = data[1]['product']
                src_prompt_pre = "The product name and description of this product are as follows: "
                src_prompt = '"\n Please summarize the description of this product and list five key words. All answers are in English and No Chinese in your answer'
                # src_title_abstract = data[1]['pro_desc']
                src_pro_desc = data[1]['pro_desc']
                node_word_input = src_prompt_pre + src_pro_desc
                input_ids = self.tokenizer(node_word_input, truncation=True, max_length=4000)["input_ids"]
                node_word_input = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                # if len(node_word_input[0]) > 3000- len(src_prompt):
                #     node_word_input = node_word_input[:3000-len(src_prompt)]
                node_word_input += src_prompt

                dst_prompt_pre = '\n The product names and descriptions of these products are provided as follows: '
                dst_prompt = "\n Please summarize the descriptions of these products. All answers are in English and No Chinese in your answer"
                # dst_title_abstract = ""
                dst_pro_desc = ""
                for neighbor_id in src_to_dst_dict[node_id]:
                    dst_pro_desc = dst_pro_desc + node_name_and_desc[neighbor_id] + '\n'

                neighbor_word_input  = dst_prompt_pre + dst_pro_desc
                input_ids = self.tokenizer(neighbor_word_input, truncation=True, max_length=4000)["input_ids"]
                neighbor_word_input = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                # if len(neighbor_word_input[0]) > 3000-len(dst_prompt):
                #     neighbor_word_input = neighbor_word_input[:3000-len(dst_prompt)]
                neighbor_word_input += dst_prompt

                try:
                    response_node, _ = self.llm.chat(self.tokenizer,
                                                            node_word_input,
                                                            history=[])
                    response_neighbor, _ = self.llm.chat(self.tokenizer,
                                                                neighbor_word_input,
                                                                history=[])
                    summary.append({
                        'node_id': node_id,
                        # 'title': title,
                        # 'product': product,
                        'response_node': response_node,
                        'response_neighbor': response_neighbor
                    })
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} product {node_id+1}")
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("CUDA out of memory error detected, skipping this batch")
                        continue
                    else:
                        continue
            else:
                continue

        summary_df = pd.DataFrame(summary)
        embeddings = torch.load(f"Baselines/GraphTranslator/data/{dataset_name}/graphsage_node_embeddings.pt").to('cpu')
        new_data = []
        for _, row in summary_df.iterrows():
            node_id = int(row['node_id'])
            embedding = np.array(embeddings[node_id].detach())
            str_array = [str(num) for num in embedding]
            str_representation = ", ".join(str_array)
            # title = node2title[row['node_id']]

            new_data.append({
                'node_id': node_id,
                'embedding':str_representation ,
                'product_summary':row['response_node'],
                'neighbour_summary':row['response_neighbor'],
                # 'title':title
                })
        summary_embeddings = pd.DataFrame(new_data)
        summary_embeddings.to_csv(f'Baselines/GraphTranslator/data/{dataset_name}/summary_embeddings_new.csv',index=False)

def main():
    setup_logging()
    init_seeds(args.distributed, args.random_seed)

    logging.info("Main arguments:")
    for k, v in args.__dict__.items():
        logging.info("{}={}".format(k, v))
    logging.info("device type: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
    dataset = args.dataset
    # load model
    if dataset == "products":
        print("loading for products:")
        model = proLLM(args)
        data, sampled_neighbors = read_dataset_products(dataset_name=dataset)
    else:
        model = LLM(args)
        data, sampled_neighbors = read_dataset(dataset_name=dataset)
    logging.info('start inference')

    np_filename = f'Baselines/GraphTranslator/data/{dataset}/{dataset}.npy'
    loaded_data_dict = np.load(np_filename, allow_pickle=True).item()

    train_ids = [int(i) for i in loaded_data_dict['train']]

    model.inference_chatglm(data, sampled_neighbors, dataset_name=dataset, train_ids=train_ids)


if __name__ == "__main__":
    main()
