import torch
import pandas as pd
import numpy as np
import re

dataset = "products"
data_path = f'Baselines/GraphTranslator/data/{dataset}/processed_data.pt'
data = torch.load(data_path)

if dataset == "products":
    data.y = data.y.squeeze()

labels = data.y.numpy()
if dataset == "cora" or dataset == "pubmed":
    titles = data.title
    titles = ["Title:" if title.strip() == "" else title for title in titles]
else:
    raw_texts = data.raw_texts
    product_names = {}
    for node_id, text in enumerate(raw_texts):
        match = re.search(r"(Product:\s*([^;]*))", text)
        product_names[node_id] = match.group(1).strip() if match else "Product:"
# print(labels[:5])
# print(data.label_texts)

# 加载 test_ids
np_filename = f'Baselines/GraphTranslator/data/{dataset}/{dataset}.npy'
loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
test_ids = [int(i) for i in loaded_data_dict['test']]

embedding_path = f'Baselines/GraphTranslator/data/{dataset}/graphsage_node_embeddings.pt'
embeddings = torch.load(embedding_path).detach().numpy()  # 将 embedding 转换为 numpy 数组

if dataset == "pubmed":
    label_map = {
        'Diabetes Mellitus Experimental': 0,
        'Diabetes Mellitus Type1': 1,
        'Diabetes Mellitus Type2': 2
    }
elif dataset == "products":
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
if dataset == "cora" or dataset == "pubmed":
    df = pd.DataFrame({
        'node_id': test_ids,
        'embedding': [", ".join(map(str, embeddings[i])) for i in test_ids],
        'product_summary': ['-'] * len(test_ids),
        'neighbour_summary': ['-'] * len(test_ids),
        'product_name': [titles[i] for i in test_ids],
        'digital_label': [labels[i] for i in test_ids]
    })
else:
    df = pd.DataFrame({
        'node_id': test_ids,
        'embedding': [", ".join(map(str, embeddings[i])) for i in test_ids],
        'product_summary': ['-'] * len(test_ids),
        'neighbour_summary': ['-'] * len(test_ids),
        'product_name': [product_names.get(i, "Product:") for i in test_ids],
        'digital_label': [labels[i] for i in test_ids]
    })

output_filename = f'Baselines/GraphTranslator/data/{dataset}/{dataset}_test.csv'
df.to_csv(output_filename, index=False)

print(f"test dataset 已保存至 {output_filename}")