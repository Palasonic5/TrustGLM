import time
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from transformers import BertTokenizer, BertModel
import numpy as np

old_data = torch.load("/scratch/xs2334/TrustGLM/Baselines/LLaGA/dataset/sup/cora_sup/processed_data.pt")

if not old_data.edge_index.is_contiguous():
    old_data.edge_index = old_data.edge_index.contiguous()

keys_to_keep = {'x', 'y', 'edge_index', 'train_id', 'val_id', 'test_id', 'num_nodes'}

data = Data()

for key in keys_to_keep:
    if hasattr(old_data, key):
        setattr(data, key, getattr(old_data, key))
    else:
        print(f"Warning: {key} not found in original data.")

dataset = "cora"
np_filename = f'Baselines/GraphTranslator/data/{dataset}/{dataset}.npy'
loaded_data_dict = np.load(np_filename, allow_pickle=True).item()

train_ids = [int(i) for i in loaded_data_dict['train']]
val_ids = [int(i) for i in loaded_data_dict['val']]
test_ids = [int(i) for i in loaded_data_dict['test']]

data.train_id = train_ids
data.val_id = val_ids
data.test_id = test_ids

print("New train_id:", data.train_id)
print("New val_id:", data.val_id)
print("New test_id:", data.test_id)

bert_node_embeddings = torch.load(f"Baselines/GraphTranslator/data/{dataset}/bert_node_embeddings.pt")
data.x = bert_node_embeddings

print("Data object created with BERT embeddings.")

train_loader = LinkNeighborLoader(
    data,
    batch_size=512,
    shuffle=True,
    neg_sampling_ratio=1.0,
    num_neighbors=[10, 10]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = data.to(device)


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

model = Net(768, 1024, 768).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)

    return total_loss / data.num_nodes


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.act(out)
        out = self.linear2(out)
        return out


def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def test():
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.edge_index)

    for epoch in range(1, 501):
        LR_model.train()
        optimizer.zero_grad()
        pred = LR_model(out[data.train_id])

        label = data.y[data.train_id].long()

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

    LR_model.eval()
    val_outputs = LR_model(out[data.val_id])
    val_acc = compute_accuracy(val_outputs, data.y[data.val_id])

    test_outputs = LR_model(out[data.test_id])
    test_acc = compute_accuracy(test_outputs, data.y[data.test_id])

    return val_acc, test_acc

times = []
best_acc = 0

for epoch in range(10):
    start = time.time()
    input_dim = 768
    output_dim = torch.max(data.y).item() + 1  # 转为 Python 整数
    LR_model = LogisticRegression(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(LR_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    loss = train()
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    val_acc, test_acc = test()
    print(f"Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    if test_acc > best_acc:
        best_acc = test_acc
        print(f"New Best Accuracy: {best_acc:.4f} at Epoch {epoch + 1}")

    times.append(time.time() - start)

model = "Baselines/GraphTranslator/Producer/inference/graphsage_models/cora_state_0110.pth"
print(f"Final Best Accuracy: {best_acc:.4f}")

out = model(data.x, data.edge_index)
print(out.shape)
torch.save(out, "Baselines/GraphTranslator/data/cora/graphsage_node_embeddings_0109.pt")

# 保存模型参数（推荐方式）
torch.save(model.state_dict(), "Baselines/GraphTranslator/Producer/inference/graphsage_models/cora_state_0109.pth")