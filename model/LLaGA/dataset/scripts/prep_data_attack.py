import copy
import random
import torch
import pandas as pd 
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import re
import json
import random
import copy
import numpy as np
import scipy.sparse as sp
import os
from transformers import AutoTokenizer
from torch_geometric.utils import degree
import argparse
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
from scipy.sparse import csr_matrix

from tqdm import trange
DEFAULT_GRAPH_PAD_ID = -500
def get_data_split(dataset,setting):
    np_filename = f'dataset/split/{setting}_{dataset}.npy'
    loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
    # Convert the numpy arrays or non-Python int types to standard Python lists of int
    train_ids = [int(i) for i in loaded_data_dict['train']]
    test_ids = [int(i) for i in loaded_data_dict['test']]
    val_ids = [int(i) for i in loaded_data_dict['val']]
    
    print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
    
    return {'train': train_ids, 'test': test_ids, 'val': val_ids}
    
def get_fix_shape_subgraph_sequence_fast(edge_list, node_idx, k_hop, sample_size, avoid_idx=None):
    assert k_hop > 0 and sample_size > 0
    neighbors = [[node_idx]]
    for t in range(k_hop):
        last_hop = neighbors[-1]
        current_hop = []
        for i in last_hop:
            if i == DEFAULT_GRAPH_PAD_ID:
                current_hop.extend([DEFAULT_GRAPH_PAD_ID]*sample_size)
                continue
            node_neighbor = copy.copy(edge_list[i])
            if t == 0 and avoid_idx is not None and  avoid_idx in node_neighbor:
                node_neighbor.remove(avoid_idx)
            if len(node_neighbor) > sample_size:
                sampled_neighbor = random.sample(node_neighbor, sample_size)
            else:
                sampled_neighbor = node_neighbor + [DEFAULT_GRAPH_PAD_ID] * (sample_size - len(node_neighbor))
            current_hop.extend(sampled_neighbor)
        neighbors.append(current_hop)
    node_sequence = [n for hop in neighbors for n in hop]
    return node_sequence

def generate_edge_list_rbcd(dataset, setting, attack):
    # data = torch.load(os.path.join(data_dir, "processed_data.pt"))
    if "grbcd" in attack:
        edge_index = torch.load(f"attack/structure_attack/grbcd/{dataset}_{setting}/global/pert_edge_index_sbert.pt", map_location=torch.device('cpu'))
    else:
        edge_index = torch.load(f"attack/structure_attack/prbcd/{dataset}_{setting}/global/pert_edge_index_sbert.pt", map_location=torch.device('cpu'))
    row, col = edge_index
    data = torch.load(f"dataset/{dataset}_{setting}/processed_data.pt")
    n = data.num_nodes
    edge_list= [[] for _ in range(n)]
    row=row.numpy()
    col=col.numpy()

    for i in trange(row.shape[0]):
        edge_list[row[i]].append(int(col[i]))

    return edge_list


def generate_edge_list_rbcd_local(dataset, setting, attack, node_id):
    # data = torch.load(os.path.join(data_dir, "processed_data.pt"))
    if "grbcd" in attack:
        edge_index = torch.load(f"attack/structure_attack/grbcd/{dataset}_{setting}/target/sbert/pert_edge_index_{node_id}.pt", map_location=torch.device('cpu'))
    else:
        edge_index = torch.load(f"attack/structure_attack/prbcd/{dataset}_{setting}/target/sbert/pert_edge_index_{node_id}.pt", map_location=torch.device('cpu'))
    row, col = edge_index
    data = torch.load(f"dataset/{dataset}_{setting}/processed_data.pt")
    n = data.num_nodes
    edge_list= [[] for _ in range(n)]
    row=row.numpy()
    col=col.numpy()

    for i in trange(row.shape[0]):
        edge_list[row[i]].append(int(col[i]))

    return edge_list

def generate_edge_list(dataset, setting, node_id, attr_type):
    """
    Load adjacency matrix from .npz file for a given node_id and convert to edge list
    
    Args:
        node_id (int): ID of the node whose adjacency matrix to load
        
    Returns:
        list: List of lists containing neighbors for each node
    """

    if node_id is not None:
        # Load .npz file for specific node
        data_dir = f"attack/structure_attack/nettack/output/edge_list/{dataset}_{setting}"
        os.makedirs(data_dir, exist_ok=True)

        edge_list_file = os.path.join(data_dir, f"edge_list_{node_id}_{attr_type}.pt")
        if os.path.exists(edge_list_file):
            print(f"Loading pre-generated edge list from {edge_list_file}")
            edge_list = torch.load(edge_list_file)
            return edge_list

        npz_file = f"attack/structure_attack/nettack/output/{dataset}_{setting}_{attr_type}/modified_adjacency_node_{node_id}.npz"
        sparse_adj = np.load(npz_file)
        
        # Get the arrays stored in the .npz file (CSR format)
        indices = sparse_adj['indices']
        indptr = sparse_adj['indptr']
        data = sparse_adj['data']
        shape = sparse_adj['shape']
    n = shape[0]  # Number of nodes
    edge_list = [[] for _ in range(n)]
    
    # Convert CSR format to edge list
    for i in range(n):
        # Get column indices for row i
        start, end = indptr[i], indptr[i + 1]
        neighbors = indices[start:end]
        
        # Add edges to edge list
        edge_list[i].extend([int(j) for j in neighbors])
    torch.save(edge_list, os.path.join(data_dir, f"edge_list_{node_id}_{attr_type}.pt"))
    return edge_list

def generate_jsonl_data(dataset, setting, k_hop=2, sample_size=10, attack=None, attr_type=None):
    data_path = f'dataset/{setting}/{dataset}_{setting}/processed_data.pt'
    data = torch.load(data_path)
    
    label_texts = data.label_texts
    print(f'{dataset} label list: {label_texts}')
    data_y = data.y
    if "rbcd_global" in attack:
        edge_list = generate_edge_list_rbcd(dataset, setting, attack)
    # edge_list = generate_edge_list(data)
    
    split_dict = get_data_split(dataset, setting)
    
    if dataset == 'amazon-photo':
        categories = "Accessories, Lighting & Studio, Underwater Photography, Video, Lenses, Film Photography, Bags & Cases, Tripods & Monopods, Digital Cameras, Video Surveillance, Flashes, Binoculars & Scopes"
        categories_num = "12"
    elif dataset == 'amazon-computers':
        categories = "Monitors, Networking Products, Laptop Accessories, Computer Accessories & Peripherals, Tablet Accessories, Tablet Replacement Parts, Servers, Computer Components, Computers & Tablets, Data Storage"
        categories_num = "10"
    elif dataset == 'amazon-sports':
        categories = "Leisure Sports & Game Room, Tennis & Racquet Sports, Golf, Airsoft & Paintball, Boating & Sailing, Swimming, Exercise & Fitness, Clothing, Sports Medicine, Other Sports, Team Sports, Hunting & Fishing, Accessories"
        categories_num = "13"
    elif dataset == 'amazon-history':
        categories = "Military, Russia, Africa, World, Australia & Oceania, Arctic & Antarctica, Historical Study & Educational Resources, Middle East, Americas, Asia, Europe, Ancient Civilizations"
        categories_num = "12"
    elif dataset == 'amazon-children':
        categories = "Literature & Fiction, History, Geography & Cultures, Science, Nature & How It Works, Biographies, Action & Adventure, Cars, Trains & Things That Go, Computers & Technology, Children's Cookbooks, Animals, Activities, Crafts & Games, Arts, Music & Photography, Early Learning, Fairy Tales, Folk Tales & Myths, Humor, Mysteries & Detectives, Growing Up & Facts of Life, Education & Reference, Science Fiction & Fantasy, Comics & Graphic Novels, Classics, Sports & Outdoors, Holidays & Celebrations, Religions"
        categories_num = "24"
    for set in ["test"]:
        node_ids = split_dict[set]
        output_file = f'Baselines/LLaGA/dataset/{setting}/{dataset}_{setting}/sampled_2_10_{set}_{attack}.jsonl'
        with open(output_file, 'w') as f:
            for node_id in node_ids:
                if attack == "nettack":
                    edge_list = generate_edge_list(dataset, setting, node_id, attr_type=attr_type)
                elif "rbcd_local" in attack:
                    edge_list = generate_edge_list_rbcd_local(dataset, setting, attack, node_id)
                # 获取图结构
                graph = get_fix_shape_subgraph_sequence_fast(edge_list, node_id, k_hop, sample_size)
                label = label_texts[int(data_y[node_id])]
                
                if dataset.startswith("amazon"):
                    # # 构建 conversation
                    conversations = [
                        {
                            "from": "human",
                            "value": f"Given a node-centered graph: <graph> where each node represents product sold in Amazon, and edges between products indicate they are purchased together, we need to classify the center node into {categories_num} classes: {categories}, please tell me which class the center node belongs to?"
                        },
                        {
                            "from": "gpt",
                            "value": label
                        }
                    ]
                        # 构建数据项
                    data_item = {
                        "id": node_id,
                        "graph": graph,
                        "conversations": conversations
                        
                    } 
                elif dataset == "cora":
                    # # 构建 conversation
                    conversations = [
                        {
                            "from": "human",
                            "value": "Given a node-centered graph: <graph>, each node represents a paper, we need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to?"
                        },
                        {
                            "from": "gpt",
                            "value": label
                        }
                    ]
                    # 构建数据项
                    data_item = {
                        "id": node_id,
                        "graph": graph,
                        "conversations": conversations
                        
                    }
                elif dataset == "pubmed":
                    conversations = [
                        {
                            "from": "human",
                            "value": "Given a node-centered graph: <graph>, each node represents a paper about Diabetes, we need to classify the center node into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2, please tell me which class the center node belongs to?"
                        },
                        {
                            "from": "gpt",
                            "value": label
                        }
                    ]
                    # 构建数据项
                    data_item = {
                        "id": node_id,
                        "graph": graph,
                        "conversations": conversations
                        
                    } 
                elif dataset == "ogbn-products":
                    conversations = [
                        {
                            "from": "human",
                            "value": f"Given a node-centered graph: <graph>, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to?"  
                        },
                        {
                            "from": "gpt",
                            "value": label
                        }
                    ]
                    # 构建数据项
                    data_item = {
                        "id": node_id,
                        "graph": graph,
                        "conversations": conversations
                        
                    }  
                elif dataset == "ogbn-arxiv":
                    conversations = [
                        {
                            "from": "human",
                            "value": "Given a node-centered graph: <graph>, we need to classify the center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please tell me which class the center node belongs to?"
                        },
                        {
                            "from": "gpt",
                            "value": label
                        }
                    ]
                    # 构建数据项
                    data_item = {
                        "id": node_id,
                        "graph": graph,
                        "conversations": conversations
                        
                    }
                    
                f.write(json.dumps(data_item) + '\n')
def main():
    # datasets = ["amazon-sports", "amazon-computers", "amazon-photo"]
    # datasets = ["ogbn-products"]
    datasets = ["pubmed"]
    # settings = ["sup","semi"]
    settings = ["sup"]
    attacks = ["prbcd_global", "prbcd_local"]
    feature_type = "sbert"
    
    # Iterate through each dataset and setting combination
    for dataset in datasets:
        for setting in settings:
            print(f"Processing dataset: {dataset}, setting: {setting}")
            for attack in attacks:

                k_hop = 2
                sample_size = 10

                    
                # Call the generate_jsonl_data function
                generate_jsonl_data(dataset, setting, k_hop, sample_size, attack, attr_type=feature_type)
            print(f"Completed dataset: {dataset}, setting: {setting}")

if __name__ == "__main__":
    main()
