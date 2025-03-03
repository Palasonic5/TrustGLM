import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from pecos.utils import smat_util
import numpy as np
import random
import scipy.sparse as sp
import os

class ProductsSemiDatasetN(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features

        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
        in_domain_classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts',
    'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing', 
    'Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
        selected_categories = in_domain_classes + categories
        print('selected')
        # random.seed(42)
        # random.shuffle(selected_categories)
        print('self shuffle', flush = True)
        print(selected_categories, flush = True)
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(selected_categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"

        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
        self.edge_indices_dict = self.generate_edge_indices_dict()

    def generate_edge_indices_dict(self):
        """
        Generate a dictionary mapping each node ID to its specific edge_index
        by dynamically loading from the storage path.
        """
        edge_indices_dict = {}

        # 获取所有节点 ID
        node_ids = self.get_idx_split()["test"]

        # 遍历所有节点，加载对应的 edge_index
        for node_id in node_ids:
            edge_index = self.load_edge_index_for_node(node_id)
            edge_indices_dict[node_id] = edge_index

        print(f"Generated edge_indices_dict for {len(edge_indices_dict)} nodes.")
        return edge_indices_dict

    def load_edge_index_for_node(self, node_id):
        """
        从存储路径加载特定节点的邻接矩阵，并转换为 edge_index 格式。
        """
        adjacency_path = f"/scratch/xs2334/TrustGLM/Graph_attack/nettack/output/ogbn-products_semi/modified_adjacency_node_{node_id}.npz"

        # 检查文件是否存在
        if not os.path.exists(adjacency_path):
            raise FileNotFoundError(f"Adjacency file for node {node_id} not found at {adjacency_path}")

        # 加载存储的稀疏邻接矩阵
        sparse_matrix = sp.load_npz(adjacency_path)

        # 转换为 edge_index 格式
        coo = sparse_matrix.tocoo()
        edge_index = np.vstack((coo.row, coo.col))  # shape: [2, num_edges]

        # 转换为 PyTorch Tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        print(f"Loaded edge_index for node {node_id}, shape: {edge_index.shape}")
        
        return edge_index


    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/semi_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]

        # sup_np_filename = f'/scratch/qz2086/graphprompter/dataset/split/sup_ogbn-products.npy'
        # loaded_data_dict = np.load(sup_np_filename, allow_pickle=True).item()
        test_ids = [int(i) for i in loaded_data_dict['test']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}


class ProductsSemiDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features

        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
        in_domain_classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts',
    'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing', 
    'Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
        selected_categories = in_domain_classes + categories
        print('selected')
        # random.seed(42)
        # random.shuffle(selected_categories)
        print('self shuffle', flush = True)
        print(selected_categories, flush = True)
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(selected_categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"

        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/semi_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]

        # sup_np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/split/sup_ogbn-products.npy'
        # loaded_data_dict = np.load(sup_np_filename, allow_pickle=True).item()
        test_ids = [int(i) for i in loaded_data_dict['test']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}
        # # Load the saved indices
        # with open('dataset/tape_products/split/train_indices.txt', 'r') as file:
        #     train_indices = [int(line.strip()) for line in file]

        # with open('dataset/tape_products/split/val_indices.txt', 'r') as file:
        #     val_indices = [int(line.strip()) for line in file]

        # with open('dataset/tape_products/split/test_indices.txt', 'r') as file:
        #     test_indices = [int(line.strip()) for line in file]
        # return {'train': train_indices, 'val': val_indices, 'test': test_indices}


class ProductsSupDatasetN(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features

        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
    #     in_domain_classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts',
    # 'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing', 
    # 'Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
        # selected_categories = in_domain_classes + categories
        # print('selected')
        # random.seed(42)
        # random.shuffle(selected_categories)
        # print('self shuffle', flush = True)
        # print(selected_categories, flush = True)
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"

        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
        self.edge_indices_dict = self.generate_edge_indices_dict()

    def generate_edge_indices_dict(self):
        """
        Generate a dictionary mapping each node ID to its specific edge_index
        by dynamically loading from the storage path.
        """
        edge_indices_dict = {}

        # 获取所有节点 ID
        node_ids = self.get_idx_split()["test"]

        # 遍历所有节点，加载对应的 edge_index
        for node_id in node_ids:
            edge_index = self.load_edge_index_for_node(node_id)
            edge_indices_dict[node_id] = edge_index

        print(f"Generated edge_indices_dict for {len(edge_indices_dict)} nodes.")
        return edge_indices_dict

    def load_edge_index_for_node(self, node_id):
        """
        从存储路径加载特定节点的邻接矩阵，并转换为 edge_index 格式。
        """
        adjacency_path = f"/scratch/xs2334/TrustGLM/Graph_attack/nettack/output/ogbn-products_sup_GIA/modified_adjacency_node_{node_id}.npz"

        # 检查文件是否存在
        if not os.path.exists(adjacency_path):
            raise FileNotFoundError(f"Adjacency file for node {node_id} not found at {adjacency_path}")

        # 加载存储的稀疏邻接矩阵
        sparse_matrix = sp.load_npz(adjacency_path)

        # 转换为 edge_index 格式
        coo = sparse_matrix.tocoo()
        edge_index = np.vstack((coo.row, coo.col))  # shape: [2, num_edges]

        # 转换为 PyTorch Tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        print(f"Loaded edge_index for node {node_id}, shape: {edge_index.shape}")
        
        return edge_index


    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]

        # sup_np_filename = f'/scratch/qz2086/graphprompter/dataset/split/sup_ogbn-products.npy'
        # loaded_data_dict = np.load(sup_np_filename, allow_pickle=True).item()
        test_ids = [int(i) for i in loaded_data_dict['test']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class ProductsSupDatasetS(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
    #     in_domain_classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts',
    # 'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing', 
    # 'Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
        # selected_categories = in_domain_classes + categories
        # print('all categories + in domain classes, placed later')
        # random.shuffle(selected_categories, seed = 42)
        # print(' shuffle', flush = True)
        random.seed(42)
        random.shuffle(categories)
        print(categories, flush = True)
        # 格式化类别为1) Category, 2) Category, ... 的形式
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}
        # # Load the saved indices
        # with open('dataset/tape_products/split/train_indices.txt', 'r') as file:
        #     train_indices = [int(line.strip()) for line in file]

        # with open('dataset/tape_products/split/val_indices.txt', 'r') as file:
        #     val_indices = [int(line.strip()) for line in file]

        # with open('dataset/tape_products/split/test_indices.txt', 'r') as file:
        #     test_indices = [int(line.strip()) for line in file]
        # return {'train': train_indices, 'val': val_indices, 'test': test_indices}

class ProductsSupDatasetLN50(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
        in_domain_classes =  ['Accessories', 'History', 'Exercise & Fitness', 'Comics & Graphic Novels', 'Europe', 'Science Fiction & Fantasy', 'Film Photography', 'Australia & Oceania', 'Cars, Trains & Things That Go', 'Airsoft & Paintball', 'Leisure Sports & Game Room', 'Servers', 'Science, Nature & How It Works', 'Religions', 'Early Learning', 'Arctic & Antarctica', 'World', 'Australia & Oceania ', 'Hunting & Fishing', 'Golf', 'Tripods & Monopods', 'Ancient Civilizations', 'Tennis & Racquet Sports', 'Computers & Technology']
    #     in_domain_classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts',
    # 'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing', 
    # 'Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
        selected_categories =  categories + in_domain_classes
        # print('all categories + in domain classes, placed later')
        # random.shuffle(selected_categories, seed = 42)
        # print(' shuffle', flush = True)
        # random.seed(42)
        # random.shuffle(categories)
        # print(categories, flush = True)
        # 格式化类别为1) Category, 2) Category, ... 的形式
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(selected_categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class ProductsSupDatasetLN100(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
        in_domain_classes = ['Accessories', 'History', 'Exercise & Fitness', 'Comics & Graphic Novels', 'Europe', 'Science Fiction & Fantasy', 'Film Photography', 'Australia & Oceania', 'Cars, Trains & Things That Go', 'Airsoft & Paintball', 'Leisure Sports & Game Room', 'Servers', 'Science, Nature & How It Works', 'Religions', 'Early Learning', 'Arctic & Antarctica', 'World', 'Australia & Oceania ', 'Hunting & Fishing', 'Golf', 'Tripods & Monopods', 'Ancient Civilizations', 'Tennis & Racquet Sports', 'Computers & Technology', 'Boating & Sailing', 'Classics', "Children's Cookbooks", 'Bags & Cases', 'Military', 'Laptop Accessories', 'Computers & Tablets', 'Video Surveillance', 'Other Sports', 'Activities, Crafts & Games', 'Swimming', 'Sports Medicine', 'Mysteries & Detectives', 'Africa', 'Sports & Outdoors', 'AccessoriesMonitors', 'Action & Adventure', 'Literature & Fiction', 'Tablet Accessories', 'Networking Products', 'Tablet Replacement Parts', 'Clothing', 'Binoculars & Scopes']
    #     in_domain_classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts',
    # 'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing', 
    # 'Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
        # selected_categories = in_domain_classes + categories
        selected_categories =  in_domain_classes + categories
        # print('all categories + in domain classes, placed later')
        # random.shuffle(selected_categories, seed = 42)
        # print(' shuffle', flush = True)
        # random.seed(42)
        # random.shuffle(categories)
        # print(categories, flush = True)
        # 格式化类别为1) Category, 2) Category, ... 的形式
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(selected_categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}
        # # Load the saved indices
        # with open('dataset/tape_products/split/train_indices.txt', 'r') as file:
        #     train_indices = [int(line.strip()) for line in file]

        # with open('dataset/tape_products/split/val_indices.txt', 'r') as file:
        #     val_indices = [int(line.strip()) for line in file]

        # with open('dataset/tape_products/split/test_indices.txt', 'r') as file:
        #     test_indices = [int(line.strip()) for line in file]
        # return {'train': train_indices, 'val': val_indices, 'test': test_indices}

class ProductsSupDatasetG(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.graph.edge_index = torch.load(f"/scratch/xs2334/TrustGLM/Graph_attack/prbcd/ogbn-products_sup/global/pert_edge_index_GIA.pt", map_location=torch.device('cpu'))
        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class ProductsSupDatasetL(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features

        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"

        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
        self.edge_indices_dict = self.generate_edge_indices_dict()

    def generate_edge_indices_dict(self):
        """
        Generate a dictionary mapping each node ID to its specific edge_index
        by dynamically loading from the storage path.
        """
        edge_indices_dict = {}

        # 获取所有节点 ID
        node_ids = self.get_idx_split()["test"]

        # 遍历所有节点，加载对应的 edge_index
        for node_id in node_ids:
            edge_index = self.load_edge_index_for_node(node_id)
            edge_indices_dict[node_id] = edge_index

        print(f"Generated edge_indices_dict for {len(edge_indices_dict)} nodes.")
        return edge_indices_dict

    def load_edge_index_for_node(self, node_id):
        """
        从存储路径加载特定节点的邻接矩阵，并转换为 edge_index 格式。
        """
        edge_index_path = f"/scratch/xs2334/TrustGLM/Graph_attack/prbcd/ogbn-products_sup/target/GIA/pert_edge_index_{node_id}.pt"
        edge_index = torch.load(edge_index_path, map_location=torch.device('cpu'))
        
        return edge_index


    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class ProductsSupDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
    #     in_domain_classes = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts',
    # 'Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing', 
    # 'Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
    #     selected_categories = in_domain_classes + categories
    #     print('all categories + in domain classes, placed later')
        # random.shuffle(selected_categories, seed = 42)
        # # print(' shuffle', flush = True)
        # print(selected_categories, flush = True)
        # 格式化类别为1) Category, 2) Category, ... 的形式
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class ProductsSupDatasetST(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
            ]
            random.shuffle(categories)
            formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(categories)])
            prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class ProductsSupDatasetLNT(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
            ]
            # 定义类列表
            classes = [
                # Children's labels
                'Literature & Fiction', 'Animals', 'Growing Up & Facts of Life', 'Humor',
                'Cars, Trains & Things That Go', 'Fairy Tales, Folk Tales & Myths',
                'Activities, Crafts & Games', 'Science Fiction & Fantasy', 'Classics',
                'Mysteries & Detectives', 'Action & Adventure', 'Geography & Cultures',
                'Education & Reference', 'Arts, Music & Photography', 'Holidays & Celebrations',
                'Science, Nature & How It Works', 'Early Learning', 'Biographies', 'History',
                "Children's Cookbooks", 'Religions', 'Sports & Outdoors', 'Comics & Graphic Novels',
                'Computers & Technology',

                # Photo labels
                'Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video',
                'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes',
                'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography',

                # History labels
                'World', 'Americas', 'Asia', 'Military', 'Europe', 'Russia', 'Africa',
                'Ancient Civilizations', 'Middle East', 'Historical Study & Educational Resources',
                'Australia & Oceania', 'Australia & Oceania ', 'Arctic & Antarctica',

                # Sports labels
                "Leisure Sports & Game Room",
                "Tennis & Racquet Sports",
                "Golf",
                "Airsoft & Paintball",
                "Boating & Sailing",
                "Swimming",
                "Exercise & Fitness",
                "Clothing",
                "Sports Medicine",
                "Other Sports",
                "Team Sports",
                "Hunting & Fishing",
                "Accessories",

                # Computers labels
                "Monitors", "Networking Products", "Laptop Accessories",
                "Computer Accessories & Peripherals", "Tablet Accessories",
                "Tablet Replacement Parts", "Servers", "Computer Components",
                "Computers & Tablets", "Data Storage"

            ]
            shuffled_classes = classes[:]
            random.shuffle(shuffled_classes)
            formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(shuffled_classes[:47]+categories)])
            prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class ProductsSupDatasetCLNT(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
            ]
            # 定义类列表
            classes = [
                # cora labels
                "Case_Based",
                "Genetic_Algorithms",
                "Neural_Networks",
                "Probabilistic_Methods",
                "Reinforcement_Learning",
                "Rule_Learning",
                "Theory",

                # pubmed labels
                "Diabetes Mellitus Experimental",
                "Diabetes Mellitus Type1",
                "Diabetes Mellitus Type2",

                "cs.NA(Numerical Analysis)", 
                "cs.MM(Multimedia)", 
                "cs.LO(Logic in Computer Science)", 
                "cs.CY(Computers and Society)", 
                "cs.CR(Cryptography and Security)", 
                "cs.DC(Distributed, Parallel, and Cluster Computing)", 
                "cs.HC(Human-Computer Interaction)", 
                "cs.CE(Computational Engineering, Finance, and Science)", 
                "cs.NI(Networking and Internet Architecture)", 
                "cs.CC(Computational Complexity)", 
                "cs.AI(Artificial Intelligence)", 
                "cs.MA(Multiagent Systems)", 
                "cs.GL(General Literature)", 
                "cs.NE(Neural and Evolutionary Computing)", 
                "cs.SC(Symbolic Computation)", 
                "cs.AR(Hardware Architecture)", 
                "cs.CV(Computer Vision and Pattern Recognition)", 
                "cs.GR(Graphics)", 
                "cs.ET(Emerging Technologies)", 
                "cs.SY(Systems and Control)", 
                "cs.CG(Computational Geometry)", 
                "cs.OH(Other Computer Science)", 
                "cs.PL(Programming Languages)", 
                "cs.SE(Software Engineering)", 
                "cs.LG(Machine Learning)", 
                "cs.SD(Sound)", 
                "cs.SI(Social and Information Networks)", 
                "cs.RO(Robotics)", 
                "cs.IT(Information Theory)", 
                "cs.PF(Performance)", 
                "cs.CL(Computational Complexity)", 
                "cs.IR(Information Retrieval)", 
                "cs.MS(Mathematical Software)", 
                "cs.FL(Formal Languages and Automata Theory)", 
                "cs.DS(Data Structures and Algorithms)", 
                "cs.OS(Operating Systems)", 
                "cs.GT(Computer Science and Game Theory)", 
                "cs.DB(Databases)", 
                "cs.DL(Digital Libraries)", 
                "cs.DM(Discrete Mathematics)"]
            shuffled_classes = classes[:]
            random.shuffle(shuffled_classes)
            formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(shuffled_classes[:47]+categories)])
            prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class ProductsSupDatasetLNC50(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
        out_domain_classes =  [
            "cs.AR(Hardware Architecture)", "cs.NE(Neural and Evolutionary Computing)", 
            "cs.CC(Computational Complexity)", "cs.MM(Multimedia)", "Reinforcement_Learning", 
            "cs.OS(Operating Systems)", "cs.CV(Computer Vision and Pattern Recognition)", 
            "Diabetes Mellitus Type2", "cs.SY(Systems and Control)", "cs.HC(Human-Computer Interaction)", 
            "cs.OH(Other Computer Science)", "cs.MA(Multiagent Systems)", "cs.LO(Logic in Computer Science)", 
            "Probabilistic_Methods", "cs.PF(Performance)", "cs.IT(Information Theory)", 
            "cs.NA(Numerical Analysis)", "cs.SC(Symbolic Computation)", "cs.SD(Sound)", 
            "Case_Based", "cs.FL(Formal Languages and Automata Theory)", 
            "cs.NI(Networking and Internet Architecture)", "cs.SE(Software Engineering)", 
            "cs.DL(Digital Libraries)"
        ]
        # selected_categories =  categories + out_domain_classes
        selected_categories =  out_domain_classes + categories
        # 格式化类别为1) Category, 2) Category, ... 的形式
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(selected_categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class ProductsSupDatasetLNC100(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        categories = [
            "Home & Kitchen", "Health & Personal Care", "Beauty", "Sports & Outdoors",
            "Books", "Patio, Lawn & Garden", "Toys & Games", "CDs & Vinyl",
            "Cell Phones & Accessories", "Grocery & Gourmet Food", "Arts, Crafts & Sewing",
            "Clothing, Shoes & Jewelry", "Electronics", "Movies & TV", "Software",
            "Video Games", "Automotive", "Pet Supplies", "Office Products",
            "Industrial & Scientific", "Musical Instruments", "Tools & Home Improvement",
            "Magazine Subscriptions", "Baby Products", "NaN", "Appliances", "Kitchen & Dining",
            "Collectibles & Fine Art", "All Beauty", "Luxury Beauty", "Amazon Fashion",
            "Computers", "All Electronics", "Purchase Circles", "MP3 Players & Accessories",
            "Gift Cards", "Office & School Supplies", "Home Improvement", "Camera & Photo",
            "GPS & Navigation", "Digital Music", "Car Electronics", "Baby", "Kindle Store",
            "Kindle Apps", "Furniture & Decor"
        ]
        out_domain_classes =  [
            "cs.AR(Hardware Architecture)", "cs.NE(Neural and Evolutionary Computing)", 
            "cs.CC(Computational Complexity)", "cs.MM(Multimedia)", "Reinforcement_Learning", 
            "cs.OS(Operating Systems)", "cs.CV(Computer Vision and Pattern Recognition)", 
            "Diabetes Mellitus Type2", "cs.SY(Systems and Control)", "cs.HC(Human-Computer Interaction)", 
            "cs.OH(Other Computer Science)", "cs.MA(Multiagent Systems)", "cs.LO(Logic in Computer Science)", 
            "Probabilistic_Methods", "cs.PF(Performance)", "cs.IT(Information Theory)", 
            "cs.NA(Numerical Analysis)", "cs.SC(Symbolic Computation)", "cs.SD(Sound)", 
            "Case_Based", "cs.FL(Formal Languages and Automata Theory)", 
            "cs.NI(Networking and Internet Architecture)", "cs.SE(Software Engineering)", 
            "cs.DL(Digital Libraries)", "cs.IR(Information Retrieval)", "cs.CG(Computational Geometry)", 
            "cs.ET(Emerging Technologies)", "cs.AI(Artificial Intelligence)", "cs.GL(General Literature)", 
            "cs.MS(Mathematical Software)", "cs.GT(Computer Science and Game Theory)", 
            "cs.SI(Social and Information Networks)", "cs.PL(Programming Languages)", 
            "cs.DS(Data Structures and Algorithms)", "cs.CY(Computers and Society)", 
            "cs.DM(Discrete Mathematics)", "cs.DB(Databases)", "Neural_Networks", 
            "cs.GR(Graphics)", "cs.RO(Robotics)", "Rule_Learning", "cs.LG(Machine Learning)", 
            "Theory", "Diabetes Mellitus Type1", "cs.CR(Cryptography and Security)", 
            "cs.DC(Distributed, Parallel, and Cluster Computing)", 
            "cs.CE(Computational Engineering, Finance, and Science)"
        ]
        # selected_categories =  categories + out_domain_classes
        selected_categories =  out_domain_classes + categories
        # 格式化类别为1) Category, 2) Category, ... 的形式
        formatted_categories = ', '.join([f"{i+1}) {cat}" for i, cat in enumerate(selected_categories)])
        self.prompt = f"\nQuestion: Which of the following category does this product belong to: {formatted_categories}? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        # self.prompt = "\nQuestion: Which of the following category does this product belong to: 1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NaN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor? Give 5 likely categories as a comma-separated list ordered from most to least likely, and provide your reasoning.\n\nAnswer:"
        self.graph_type = 'Product co-purchasing network'
        self.num_features = 768
        self.num_classes = 47
        print(f'label mapping: {self.graph.label_texts}')
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):

            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-products/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-products.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

if __name__ == '__main__':
    dataset = ProductsDataset()

    print(dataset.graph)
    print(dataset.prompt)
    print(json.dumps(dataset[22], indent=4))

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
