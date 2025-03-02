import time
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import numpy as np
import os

dataset = "arxiv"

# Load BERT embeddings
bert_node_embeddings = torch.load(f"Baselines/GraphTranslator/data/{dataset}/bert_node_embeddings.pt")

# Load shared adjacency matrix
edge_index_path = f"Graph_attack/prbcd/ogbn-{dataset}_sup/global/pert_edge_index_bert.pt"
edge_index = torch.load(edge_index_path, map_location=torch.device('cpu'))

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

# Load test IDs
np_filename = f'Baselines/GraphTranslator/data/{dataset}/{dataset}.npy'
loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
test_ids = [int(i) for i in loaded_data_dict['test']]

# Directory to save node embeddings
output_dir = f"Baselines/GraphTranslator/data/{dataset}/prbcd_global_test_emb"
os.makedirs(output_dir, exist_ok=True)

# Run inference for test nodes only
with torch.no_grad():
    out = model(bert_node_embeddings.to(device), edge_index.to(device))

# Save output for each test node
for node_id in test_ids:
    node_embedding = out[node_id].cpu()
    output_path = os.path.join(output_dir, f"node_{node_id}_embedding.pt")
    torch.save(node_embedding, output_path)
    print(f"Node {node_id} embedding saved to {output_path}")