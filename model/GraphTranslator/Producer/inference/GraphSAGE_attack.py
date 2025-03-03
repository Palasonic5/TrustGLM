import argparse
import time
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import numpy as np
import os
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser(description="Run inference on graph attacks")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., arxiv, ogbn-products, pubmed)")
parser.add_argument("--attack", type=str, required=True, choices=["nettack", "prbcd_local", "prbcd_global"], help="Attack type")
args = parser.parse_args()

dataset = args.dataset
attack = args.attack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_node_embeddings = torch.load(f"model/GraphTranslator/data/{dataset}/bert_node_embeddings.pt")

class Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model_path = f"model/GraphTranslator/Producer/inference/graphsage_models/{dataset}_state.pth"
model = Net(768, 1024, 768).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

if attack == "nettack":
    adj_dir = f"structure_attack/nettack/output/{dataset}_sup_bert"
    output_dir = f"model/GraphTranslator/data/{dataset}/nettack_test_emb"
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(adj_dir):
        if file_name.startswith("modified_adjacency_node_") and file_name.endswith(".npz"):
            node_id = int(file_name.split("_")[-1].split(".")[0])
            sparse_adj = np.load(os.path.join(adj_dir, file_name), allow_pickle=True)
            adj_matrix = csr_matrix((sparse_adj['data'], sparse_adj['indices'], sparse_adj['indptr']), shape=sparse_adj['shape'])

            src, dst = adj_matrix.nonzero()
            edge_index = torch.tensor([src, dst], dtype=torch.long)

            data = Data(x=bert_node_embeddings, edge_index=edge_index)
            data = data.to(device)

            with torch.no_grad():
                out = model(data.x, data.edge_index)
                node_embedding = out[node_id].cpu()

            output_path = os.path.join(output_dir, f"node_{node_id}_embedding.pt")
            torch.save(node_embedding, output_path)
            print(f"Nettack - Node {node_id} embedding saved to {output_path}")

elif attack == "prbcd_local":
    edge_index_dir = f"structure_attack/prbcd/{dataset}_sup/target/bert"
    output_dir = f"model/GraphTranslator/data/{dataset}/prbcd_local_test_emb"
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(edge_index_dir):
        if file_name.startswith("pert_edge_index_") and file_name.endswith(".pt"):
            node_id = int(file_name.split("_")[-1].split(".")[0])
            edge_index = torch.load(f"{edge_index_dir}/{file_name}", map_location=torch.device('cpu'))

            data = Data(x=bert_node_embeddings, edge_index=edge_index)
            data = data.to(device)

            with torch.no_grad():
                out = model(data.x, data.edge_index)
                node_embedding = out[node_id].cpu()

            output_path = os.path.join(output_dir, f"node_{node_id}_embedding.pt")
            torch.save(node_embedding, output_path)
            print(f"Prbcd_Local - Node {node_id} embedding saved to {output_path}")

elif attack == "prbcd_global":
    edge_index_path = f"structure_attack/prbcd/ogbn-{dataset}_sup/global/pert_edge_index_bert.pt"
    edge_index = torch.load(edge_index_path, map_location=torch.device('cpu'))

    np_filename = f'model/GraphTranslator/data/{dataset}/{dataset}.npy'
    loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
    test_ids = [int(i) for i in loaded_data_dict['test']]

    output_dir = f"model/GraphTranslator/data/{dataset}/prbcd_global_test_emb"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        out = model(bert_node_embeddings.to(device), edge_index.to(device))

    for node_id in test_ids:
        node_embedding = out[node_id].cpu()
        output_path = os.path.join(output_dir, f"node_{node_id}_embedding.pt")
        torch.save(node_embedding, output_path)
        print(f"Prbcd_Global - Node {node_id} embedding saved to {output_path}")
