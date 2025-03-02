import time
# from ogb.nodeproppred import PygNodePropPredDataset
# import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm

dataset = "arxiv"
old_data = torch.load(f"Baselines/GraphTranslator/data/{dataset}/processed_data.pt")

# 确保 edge_index 是连续的
if not old_data.edge_index.is_contiguous():
    old_data.edge_index = old_data.edge_index.contiguous()

# 使用本地的 BERT 模型和分词器
bert_model_path = "Baselines/GraphTranslator/Translator/models/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_path)
model = BertModel.from_pretrained(bert_model_path)

print("Encoding node texts using BERT...")
raw_texts = old_data.raw_texts
embeddings = []

# 对每个节点的文本进行编码
for text in raw_texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        # 使用 [CLS] token 的输出作为嵌入
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(embedding)

# 转换为 numpy 数组并保存
embeddings = np.array(embeddings)
torch.save(torch.tensor(embeddings), f"Baselines/GraphTranslator/data/{dataset}/bert_node_embeddings.pt")

print("BERT embeddings saved successfully.")

# 保留的属性
keys_to_keep = {'x', 'y', 'edge_index', 'train_id', 'val_id', 'test_id', 'num_nodes'}

# 创建一个新的数据对象，只保留指定的属性
data = Data()

for key in keys_to_keep:
    if hasattr(old_data, key):
        setattr(data, key, getattr(old_data, key))
    else:
        print(f"Warning: {key} not found in original data.")

# 加载新的 train, val, test id
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

# 加载保存的嵌入并设置为 data.x
bert_node_embeddings = torch.load(f"Baselines/GraphTranslator/data/{dataset}/bert_node_embeddings.pt")
# 设置 data.x 为 BERT 编码的嵌入
data.x = bert_node_embeddings

print("Data object created with BERT embeddings.")

if dataset == "products":
    data.y = data.y.squeeze()
print(data)

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

for epoch in tqdm(range(10), desc="Epoch Progress", leave=True):
    start = time.time()

    input_dim = 768
    output_dim = torch.max(data.y).item() + 1
    

    loss = train()
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    if epoch == 9:
        LR_model = LogisticRegression(input_dim, output_dim).to(device)
        optimizer = torch.optim.Adam(LR_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        val_acc, test_acc = test()
        print(f"Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    times.append(time.time() - start)

print(f"Final Best Accuracy: {best_acc:.4f}")

out = model(data.x, data.edge_index)
print(out.shape)
torch.save(out, f"Baselines/GraphTranslator/data/{dataset}/graphsage_node_embeddings.pt")

torch.save(model.state_dict(), f"Baselines/GraphTranslator/Producer/inference/graphsage_models/{dataset}_state.pth")