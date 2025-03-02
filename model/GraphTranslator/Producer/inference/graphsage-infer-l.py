import time
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import numpy as np
import os
from scipy.sparse import csr_matrix

dataset = "pubmed"

# Load BERT embeddings
bert_node_embeddings = torch.load(f"Baselines/GraphTranslator/data/{dataset}/bert_node_embeddings.pt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Load pre-trained model
model_path = f"Baselines/GraphTranslator/Producer/inference/graphsage_models/{dataset}_state.pth"
model = Net(768, 1024, 768).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Directory containing modified adjacency matrices
edge_index_dir = f"Graph_attack/prbcd/{dataset}_sup/target/bert"
output_dir = f"Baselines/GraphTranslator/data/{dataset}/prbcd_local_test_emb"
os.makedirs(output_dir, exist_ok=True)

# Run inference for each test node
for file_name in os.listdir(edge_index_dir):
    if file_name.startswith("pert_edge_index_") and file_name.endswith(".pt"):
        node_id = int(file_name.split("_")[-1].split(".")[0])
        edge_index = torch.load(f"{edge_index_dir}/{file_name}", map_location=torch.device('cpu'))

        # Create data object for the node
        data = Data(x=bert_node_embeddings, edge_index=edge_index)
        data = data.to(device)

        # Run inference
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            node_embedding = out[node_id].cpu()

        # Save output for the node
        output_path = os.path.join(output_dir, f"node_{node_id}_embedding.pt")
        torch.save(node_embedding, output_path)
        print(f"Node {node_id} embedding saved to {output_path}")