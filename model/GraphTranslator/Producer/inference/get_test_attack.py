import torch
import pandas as pd
import numpy as np
import os
import re
import argparse

parser = argparse.ArgumentParser(description="Process dataset and attack method.")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., cora, pubmed, ogbn-products)")
parser.add_argument("--attack", type=str, required=True, help="Attack method name (e.g., prbcd_global)")

args = parser.parse_args()
dataset = args.dataset
attack = args.attack

# Paths
data_path = f'Baselines/GraphTranslator/data/{dataset}/processed_data.pt'
np_filename = f'Baselines/GraphTranslator/data/{dataset}/{dataset}.npy'
embedding_dir = f'Baselines/GraphTranslator/data/{dataset}/{attack}_test_emb'
output_filename = f'Baselines/GraphTranslator/data/{dataset}/{dataset}_test_{attack}.csv'

# Load data and labels
data = torch.load(data_path)
if "products" in dataset:
    data.y = data.y.reshape(-1)
labels = data.y.numpy()

# Load test IDs
loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
test_ids = [int(i) for i in loaded_data_dict['test']]

# Prepare label map
if dataset == "cora" or dataset == "pubmed" or "arxiv" in dataset:
    titles = data.title
    titles = ["Title:" if title.strip() == "" else title for title in titles]
    if dataset == "cora":
        label_map = {
            "Case_Based": 0,
            "Genetic_Algorithms": 1,
            "Neural_Networks": 2,
            "Probabilistic_Methods": 3,
            "Reinforcement_Learning": 4,
            "Rule_Learning": 5,
            "Theory": 6
        }
    elif dataset == "pubmed":
        label_map = {
            'Diabetes Mellitus Experimental': 0,
            'Diabetes Mellitus Type1': 1,
            'Diabetes Mellitus Type2': 2
        }
    elif "arxiv" in dataset:
        label_map = {
            "cs.NA(Numerical Analysis)": 0,
            "cs.MM(Multimedia)": 1,
            "cs.LO(Logic in Computer Science)": 2,
            "cs.CY(Computers and Society)": 3,
            "cs.CR(Cryptography and Security)": 4,
            "cs.DC(Distributed, Parallel, and Cluster Computing)": 5,
            "cs.HC(Human-Computer Interaction)": 6,
            "cs.CE(Computational Engineering, Finance, and Science)": 7,
            "cs.NI(Networking and Internet Architecture)": 8,
            "cs.CC(Computational Complexity)": 9,
            "cs.AI(Artificial Intelligence)": 10,
            "cs.MA(Multiagent Systems)": 11,
            "cs.GL(General Literature)": 12,
            "cs.NE(Neural and Evolutionary Computing)": 13,
            "cs.SC(Symbolic Computation)": 14,
            "cs.AR(Hardware Architecture)": 15,
            "cs.CV(Computer Vision and Pattern Recognition)": 16,
            "cs.GR(Graphics)": 17,
            "cs.ET(Emerging Technologies)": 18,
            "cs.SY(Systems and Control)": 19,
            "cs.CG(Computational Geometry)": 20,
            "cs.OH(Other Computer Science)": 21,
            "cs.PL(Programming Languages)": 22,
            "cs.SE(Software Engineering)": 23,
            "cs.LG(Machine Learning)": 24,
            "cs.SD(Sound)": 25,
            "cs.SI(Social and Information Networks)": 26,
            "cs.RO(Robotics)": 27,
            "cs.IT(Information Theory)": 28,
            "cs.PF(Performance)": 29,
            "cs.CL(Computational Complexity)": 30,
            "cs.IR(Information Retrieval)": 31,
            "cs.MS(Mathematical Software)": 32,
            "cs.FL(Formal Languages and Automata Theory)": 33,
            "cs.DS(Data Structures and Algorithms)": 34,
            "cs.OS(Operating Systems)": 35,
            "cs.GT(Computer Science and Game Theory)": 36,
            "cs.DB(Databases)": 37,
            "cs.DL(Digital Libraries)": 38,
            "cs.DM(Discrete Mathematics)": 39
        }

    # Load embeddings for test nodes
    embeddings = []
    for node_id in test_ids:
        embedding_path = os.path.join(embedding_dir, f"node_{node_id}_embedding.pt")
        node_embedding = torch.load(embedding_path).numpy()
        embeddings.append(", ".join(map(str, node_embedding)))

    # Create DataFrame
    df = pd.DataFrame({
        'node_id': test_ids,
        'embedding': embeddings,
        'paper_summary': ['-'] * len(test_ids),
        'citepapers_summary': ['-'] * len(test_ids),
        'title': [titles[i] for i in test_ids],
        'digital_label': [labels[i] for i in test_ids]
    })
elif "products" in dataset:
    label_map = {
        'Home & Kitchen': 0, 'Health & Personal Care': 1, 'Beauty': 2, 'Sports & Outdoors': 3, 'Books': 4,
        'Patio, Lawn & Garden': 5, 'Toys & Games': 6, 'CDs & Vinyl': 7, 'Cell Phones & Accessories': 8, 'Grocery & Gourmet Food': 9,
        'Arts, Crafts & Sewing': 10, 'Clothing, Shoes & Jewelry': 11, 'Electronics': 12, 'Movies & TV': 13, 'Software': 14, 'Video Games': 15,
        'Automotive': 16, 'Pet Supplies': 17, 'Office Products': 18, 'Industrial & Scientific': 19, 'Musical Instruments': 20,
        'Tools & Home Improvement': 21, 'Magazine Subscriptions': 22, 'Baby Products': 23, 'label 25': 24, 'Appliances': 25,
        'Kitchen & Dining': 26, 'Collectibles & Fine Art': 27, 'All Beauty': 28, 'Luxury Beauty': 29, 'Amazon Fashion': 30, 'Computers': 31,
        'All Electronics': 32, 'Purchase Circles': 33, 'MP3 Players & Accessories': 34, 'Gift Cards': 35, 'Office & School Supplies': 36,
        'Home Improvement': 37, 'Camera & Photo': 38, 'GPS & Navigation': 39, 'Digital Music': 40, 'Car Electronics': 41, 'Baby': 42,
        'Kindle Store': 43, 'Buy a Kindle': 44, 'Furniture & Decor': 45, '#508510': 46
        }
       # Load embeddings for test nodes
    embeddings = []
    for node_id in test_ids:
        embedding_path = os.path.join(embedding_dir, f"node_{node_id}_embedding.pt")
        node_embedding = torch.load(embedding_path).numpy()
        embeddings.append(", ".join(map(str, node_embedding)))
    
    raw_texts = data.raw_texts
    # 解析 product_name，保持 "Product:" 作为前缀
    node_to_product = {}
    for node_id, text in enumerate(raw_texts):
        match = re.search(r"(Product:\s*([^;]*))", text)  # 允许 "Product:" 但后面可能没有内容
        product_name = match.group(1).strip() if match else "Product:"  # 保持 "Product:" 作为前缀
        node_to_product[node_id] = product_name

    # Create DataFrame
    df = pd.DataFrame({
        'node_id': test_ids,
        'embedding': embeddings,
        'paper_summary': ['-'] * len(test_ids),
        'citepapers_summary': ['-'] * len(test_ids),
        'product_name': [node_to_product.get(node_id, "Product:") for node_id in test_ids],
        'digital_label': [labels[i] for i in test_ids]
    })

# Save to CSV
df.to_csv(output_filename, index=False)
print(f"{dataset}_test.csv 已保存至 {output_filename}")
