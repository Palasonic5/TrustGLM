import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from pecos.utils import smat_util
import numpy as np
import scipy.sparse as sp
import os
import random

class CoraSemiDatasetN(Dataset):
    def __init__(self):
        super().__init__()

        # Load the graph data
        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "Please predict the most appropriate category for the paper. Choose from the following categories:\nRule Learning\nNeural Networks\nCase Based\nGenetic Algorithms\nTheory\nReinforcement Learning\nProbabilistic Methods\n\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features

        # # Load and set the new edge_index
        # self.graph.edge_index = self.load_modified_edge_index(
        #     '/scratch/xs2334/TrustGLM/Graph_attack/gnn-meta-attack/output/cora_semi/modified_adjacency_new.npy'
        # )
        self.edge_indices_dict = self.generate_edge_indices_dict()

        self.num_features = 768
        self.num_classes = 7
        print(f'label mapping: {self.graph.label_texts}')

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
        adjacency_path = f"/scratch/xs2334/TrustGLM/Graph_attack/nettack/output/cora_semi_GIA/modified_adjacency_node_{node_id}.npz"

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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/semi_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}


class CoraSemiDatasetM(Dataset):
    def __init__(self):
        super().__init__()

        # Load the graph data
        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "Please predict the most appropriate category for the paper. Choose from the following categories:\nRule Learning\nNeural Networks\nCase Based\nGenetic Algorithms\nTheory\nReinforcement Learning\nProbabilistic Methods\n\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features

        # Load and set the new edge_index
        self.graph.edge_index = self.load_modified_edge_index(
            '/scratch/xs2334/TrustGLM/Graph_attack/gnn-meta-attack/output/cora_semi/modified_adjacency_new.npy'
        )

        self.num_features = 768
        self.num_classes = 7
        print(f'label mapping: {self.graph.label_texts}')

    def load_modified_edge_index(self, adjacency_path):
        """
        Load the modified adjacency matrix and convert it to edge_index format.
        """
        # Load the adjacency matrix
        adjacency_matrix = np.load(adjacency_path)

        # Ensure it's a sparse matrix
        sparse_adj = sp.coo_matrix(adjacency_matrix)

        # Convert to edge_index format
        edge_index = np.vstack((sparse_adj.row, sparse_adj.col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        print(f"Loaded edge_index from {adjacency_path}, shape: {edge_index.shape}")
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/semi_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}


class CoraSemiDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "Please predict the most appropriate category for the paper. Choose from the following categories:\nRule Learning\nNeural Networks\nCase Based\nGenetic Algorithms\nTheory\nReinforcement Learning\nProbabilistic Methods\n\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 7
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/semi_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]

        # 用semi的split试试看
        # sup_np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        # loaded_data_dict = np.load(sup_np_filename, allow_pickle=True).item()
        test_ids = [int(i) for i in loaded_data_dict['test']]

        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

        # # Load the saved indices
        # with open('dataset/tape_cora/split/train_indices.txt', 'r') as file:
        #     train_indices = [int(line.strip()) for line in file]

        # with open('dataset/tape_cora/split/val_indices.txt', 'r') as file:
        #     val_indices = [int(line.strip()) for line in file]

        # with open('dataset/tape_cora/split/test_indices.txt', 'r') as file:
        #     test_indices = [int(line.strip()) for line in file]
        # return {'train': train_indices, 'val': val_indices, 'test': test_indices}


class CoraSupDatasetN(Dataset):
    def __init__(self):
        super().__init__()

        # Load the graph data
        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "Please predict the most appropriate category for the paper. Choose from the following categories:\nRule Learning\nNeural Networks\nCase Based\nGenetic Algorithms\nTheory\nReinforcement Learning\nProbabilistic Methods\n\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features

        # # Load and set the new edge_index
        # self.graph.edge_index = self.load_modified_edge_index(
        #     '/scratch/xs2334/TrustGLM/Graph_attack/gnn-meta-attack/output/cora_semi/modified_adjacency_new.npy'
        # )
        self.edge_indices_dict = self.generate_edge_indices_dict()

        self.num_features = 768
        self.num_classes = 7
        print(f'label mapping: {self.graph.label_texts}')

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
        adjacency_path = f"/scratch/xs2334/TrustGLM/Graph_attack/nettack/output/cora_sup_GIA/modified_adjacency_node_{node_id}.npz"

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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class CoraSupDatasetM(Dataset):
    def __init__(self):
        super().__init__()

        # Load the graph data
        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "Please predict the most appropriate category for the paper. Choose from the following categories:\nRule Learning\nNeural Networks\nCase Based\nGenetic Algorithms\nTheory\nReinforcement Learning\nProbabilistic Methods\n\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features

        # Load and set the new edge_index
        self.graph.edge_index = self.load_modified_edge_index(
            '/scratch/xs2334/TrustGLM/Graph_attack/gnn-meta-attack/output/cora_sup_GIA/modified_adjacency.npy'
        )

        self.num_features = 768
        self.num_classes = 7
        print(f'label mapping: {self.graph.label_texts}')

    def load_modified_edge_index(self, adjacency_path):
        """
        Load the modified adjacency matrix and convert it to edge_index format.
        """
        # Load the adjacency matrix
        adjacency_matrix = np.load(adjacency_path)

        # Ensure it's a sparse matrix
        sparse_adj = sp.coo_matrix(adjacency_matrix)

        # Convert to edge_index format
        edge_index = np.vstack((sparse_adj.row, sparse_adj.col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        print(f"Loaded edge_index from {adjacency_path}, shape: {edge_index.shape}")
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}


class CoraSupDatasetS(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        categories = [
                        "Rule Learning",
                        "Neural Networks",
                        "Case Based",
                        "Genetic Algorithms",
                        "Theory",
                        "Reinforcement Learning",
                        "Probabilistic Methods"
                    ]
        random.seed(42)
        random.shuffle(categories)
        str_categories = "\n".join(categories)
        self.prompt = f"Please predict the most appropriate category for the paper. Choose from the following categories:\n{str_categories}\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 7
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class CoraSupDatasetLN50(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        categories = [
                        "Rule Learning",
                        "Neural Networks",
                        "Case Based",
                        "Genetic Algorithms",
                        "Theory",
                        "Reinforcement Learning",
                        "Probabilistic Methods"
                    ]
        # random.seed(42)
        # random.shuffle(categories)
        in_domain_class = ["Hydrology", "cs.GL (General Literature)", "Materials Science", "Analytical Chemistry"]
        str_categories = "\n".join(categories + in_domain_class)
        self.prompt = f"Please predict the most appropriate category for the paper. Choose from the following categories:\n{str_categories}\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 7
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class CoraSupDatasetLN100(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        categories = [
                        "Rule Learning",
                        "Neural Networks",
                        "Case Based",
                        "Genetic Algorithms",
                        "Theory",
                        "Reinforcement Learning",
                        "Probabilistic Methods"
                    ]
        # random.seed(42)
        # random.shuffle(categories)
        in_domain_class = ["Hydrology", "cs.GL (General Literature)", "Materials Science", "Analytical Chemistry", "cs.PF (Performance)", "cs.CC (Computational Complexity)", "Physical Chemistry"]
        str_categories = "\n".join(categories + in_domain_class)
        self.prompt = f"Please predict the most appropriate category for the paper. Choose from the following categories:\n{str_categories}\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 7
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class CoraSupDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "Please predict the most appropriate category for the paper. Choose from the following categories:\nRule Learning\nNeural Networks\nCase Based\nGenetic Algorithms\nTheory\nReinforcement Learning\nProbabilistic Methods\n\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 7
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class CoraSupDatasetST(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        # self.prompt = "Please predict the most appropriate category for the paper. Choose from the following categories:\nRule Learning\nNeural Networks\nCase Based\nGenetic Algorithms\nTheory\nReinforcement Learning\nProbabilistic Methods\n\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 7
        print(f'label mapping: {self.graph.label_texts}')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            categories = [
                "Rule Learning",
                "Neural Networks",
                "Case Based",
                "Genetic Algorithms",
                "Theory",
                "Reinforcement Learning",
                "Probabilistic Methods"
            ]
            # random.seed(42)
            random.shuffle(categories)
            str_categories = "\n".join(categories)
            prompt = f"Please predict the most appropriate category for the paper. Choose from the following categories:\n{str_categories}\nAnswer:"
            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class CoraSupDatasetLNT(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        # self.prompt = "Please predict the most appropriate category for the paper. Choose from the following categories:\nRule Learning\nNeural Networks\nCase Based\nGenetic Algorithms\nTheory\nReinforcement Learning\nProbabilistic Methods\n\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 7
        print(f'label mapping: {self.graph.label_texts}')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            categories = [
                "Rule Learning",
                "Neural Networks",
                "Case Based",
                "Genetic Algorithms",
                "Theory",
                "Reinforcement Learning",
                "Probabilistic Methods"
            ]
            # 定义类列表
            classes = [

                # pubmed labels
                "Diabetes Mellitus Experimental",
                "Diabetes Mellitus Type1",
                "Diabetes Mellitus Type2",

                # arxiv labels
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
                "cs.DM(Discrete Mathematics)",
                # mag labels
                "Biochemistry",
                "Astrophysics",
                "Ecology",
                "Environmental Science",
                "Agriculture",
                "Marine Biology",
                "Oceanography",
                "Materials Science",
                "Pharmacology",
                "Organic Chemistry",
                "Analytical Chemistry",
                "Physical Chemistry",
                "Paleontology",
                "Geology",
                "Meteorology",
                "Hydrology",
                "Public Health",
                "Epidemiology",
                "Immunology",
                "Pathology",
                "Cardiology",
                "Gastroenterology",
                "Dermatology",
                "Pediatrics",
                "Orthopedics",
                "Neurosurgery",
                "Radiology",
                "Medical Imaging",
                "Molecular Biology",
                "Cancer Research",
                "Genetic Engineering",
                "Genomics",
                "Proteomics",
                "Tissue Engineering",
                "Veterinary Science",
                "Sociology",
                "Anthropology",
                "Archaeology",
                "Linguistics",
                "Philosophy"
            ]
            # random.seed(42)
            shuffled_classes = classes[:]
            random.shuffle(shuffled_classes)
            str_categories = "\n".join(shuffled_classes[:7]+categories)
            prompt = f"Please predict the most appropriate category for the paper. Choose from the following categories:\n{str_categories}\nAnswer:"
            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class CoraSupDatasetCLNT(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        # self.prompt = "Please predict the most appropriate category for the paper. Choose from the following categories:\nRule Learning\nNeural Networks\nCase Based\nGenetic Algorithms\nTheory\nReinforcement Learning\nProbabilistic Methods\n\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 7
        print(f'label mapping: {self.graph.label_texts}')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            categories = [
                "Rule Learning",
                "Neural Networks",
                "Case Based",
                "Genetic Algorithms",
                "Theory",
                "Reinforcement Learning",
                "Probabilistic Methods"
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
                "Accessories"

                # Computers labels
                "Monitors", "Networking Products", "Laptop Accessories",
                "Computer Accessories & Peripherals", "Tablet Accessories",
                "Tablet Replacement Parts", "Servers", "Computer Components",
                "Computers & Tablets", "Data Storage"

                # Products labels
                "Home & Kitchen", 
                "Health & Personal Care", 
                "Beauty", 
                "Sports & Outdoors", 
                "Books", 
                "Patio, Lawn & Garden", 
                "Toys & Games", 
                "CDs & Vinyl", 
                "Cell Phones & Accessories", 
                "Grocery & Gourmet Food", 
                "Arts, Crafts & Sewing", 
                "Clothing, Shoes & Jewelry", 
                "Electronics", 
                "Movies & TV", 
                "Software", 
                "Video Games", 
                "Automotive", 
                "Pet Supplies", 
                "Office Products", 
                "Industrial & Scientific", 
                "Musical Instruments", 
                "Tools & Home Improvement", 
                "Magazine Subscriptions", 
                "Baby Products", 
                "label 25", 
                "Appliances", 
                "Kitchen & Dining", 
                "Collectibles & Fine Art", 
                "All Beauty", 
                "Luxury Beauty", 
                "Amazon Fashion", 
                "Computers", 
                "All Electronics", 
                "Purchase Circles", 
                "MP3 Players & Accessories", 
                "Gift Cards", 
                "Office & School Supplies", 
                "Home Improvement", 
                "Camera & Photo", 
                "GPS & Navigation", 
                "Digital Music", 
                "Car Electronics", 
                "Baby", 
                "Kindle Store", 
                "Buy a Kindle", 
                "Furniture & D&#233;cor,", 
                "#508510"
            ]
            # random.seed(42)
            shuffled_classes = classes[:]
            random.shuffle(shuffled_classes)
            str_categories = "\n".join(shuffled_classes[:7]+categories)
            prompt = f"Please predict the most appropriate category for the paper. Choose from the following categories:\n{str_categories}\nAnswer:"
            return {
                'id': index,
                'label': self.graph.label_texts[int(self.graph.y[index])],
                'desc': self.text[index],
                'question': prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class CoraSupDatasetG(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "Please predict the most appropriate category for the paper. Choose from the following categories:\nRule Learning\nNeural Networks\nCase Based\nGenetic Algorithms\nTheory\nReinforcement Learning\nProbabilistic Methods\n\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.graph.edge_index = torch.load(f"/scratch/xs2334/TrustGLM/Graph_attack/prbcd/cora_sup/global/pert_edge_index_GIA.pt", map_location=torch.device('cpu'))
        self.num_features = 768
        self.num_classes = 7
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class CoraSupDatasetL(Dataset):
    def __init__(self):
        super().__init__()

        # Load the graph data
        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "Please predict the most appropriate category for the paper. Choose from the following categories:\nRule Learning\nNeural Networks\nCase Based\nGenetic Algorithms\nTheory\nReinforcement Learning\nProbabilistic Methods\n\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.edge_indices_dict = self.generate_edge_indices_dict()

        self.num_features = 768
        self.num_classes = 7
        print(f'label mapping: {self.graph.label_texts}')

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
        edge_index_path = f"/scratch/xs2334/TrustGLM/Graph_attack/prbcd/cora_sup/target/GIA/pert_edge_index_{node_id}.pt"
        edge_index = torch.load(edge_index_path, map_location=torch.device('cpu'))
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        train_ids = [int(i) for i in loaded_data_dict['train']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class CoraSupDatasetLNC50(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        categories = [
                        "Rule Learning",
                        "Neural Networks",
                        "Case Based",
                        "Genetic Algorithms",
                        "Theory",
                        "Reinforcement Learning",
                        "Probabilistic Methods"
                    ]
        out_domain_class = ["Computer Components", "Car Electronics", "Electronics", "Tennis & Racquet Sports"]
        # str_categories = "\n".join(categories + out_domain_class)
        str_categories = "\n".join(out_domain_class + categories)
        self.prompt = f"Please predict the most appropriate category for the paper. Choose from the following categories:\n{str_categories}\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 7
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class CoraSupDatasetLNC100(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        categories = [
                        "Rule Learning",
                        "Neural Networks",
                        "Case Based",
                        "Genetic Algorithms",
                        "Theory",
                        "Reinforcement Learning",
                        "Probabilistic Methods"
                    ]
        out_domain_class = ["Computer Components", "Car Electronics", "Electronics", "Tennis & Racquet Sports", "Russia", "Early Learning", "Software"]
        # str_categories = "\n".join(categories + out_domain_class)
        str_categories = "\n".join(out_domain_class + categories)
        self.prompt = f"Please predict the most appropriate category for the paper. Choose from the following categories:\n{str_categories}\nAnswer:"
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 7
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/cora/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_cora.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

if __name__ == '__main__':
    dataset = CoraDataset()

    print(dataset.graph)
    print(dataset.prompt)
    print(json.dumps(dataset[0], indent=4))

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
