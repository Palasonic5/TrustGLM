import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from pecos.utils import smat_util
import numpy as np

class ArxivSupDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "\nQuestion: Which arXiv CS sub-category does this paper belong to? Give your answer in the form \'cs.XX\'.\nAnswer: "
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-arxiv/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 40
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-arxiv/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-arxiv.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class ArxivSupDatasetS(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "\nQuestion: Which arXiv CS sub-category does this paper belong to? Give your answer in the form \'cs.XX\'.\nAnswer: "
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-arxiv/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 40
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-arxiv/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-arxiv.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class ArxivSupDatasetG(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "\nQuestion: Which arXiv CS sub-category does this paper belong to? Give your answer in the form \'cs.XX\'.\nAnswer: "
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-arxiv/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.graph.edge_index = torch.load(f"/scratch/xs2334/TrustGLM/Graph_attack/prbcd/ogbn-arxiv_sup/global/pert_edge_index_GIA.pt", map_location=torch.device('cpu'))
        self.num_features = 768
        self.num_classes = 40
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-arxiv/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-arxiv.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}

class ArxivSupDatasetL(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "\nQuestion: Which arXiv CS sub-category does this paper belong to? Give your answer in the form \'cs.XX\'.\nAnswer: "
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-arxiv/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 40
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
        edge_index_path = f"/scratch/xs2334/TrustGLM/Graph_attack/prbcd/ogbn-arxiv_sup/target/GIA/pert_edge_index_{node_id}.pt"
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
        return ['/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/ogbn-arxiv/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/xs2334/TrustGLM/Baselines/graphprompter/dataset/split/sup_ogbn-arxiv.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]
        test_ids = [int(i) for i in loaded_data_dict['test']]
        val_ids = [int(i) for i in loaded_data_dict['val']]
        
        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}


class ArxivSemiDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = self.graph.raw_texts
        self.prompt = "\nQuestion: Which arXiv CS sub-category does this paper belong to? Give your answer in the form \'cs.XX\'.\nAnswer: "
        self.graph_type = 'Text Attributed Graph'
        feature_path = '/scratch/ys6310/graphprompter/dataset/ogbn-arxiv/GIA.emb'
        features = torch.from_numpy(smat_util.load_matrix(feature_path).astype(np.float32))
        self.graph.x = features
        self.num_features = 768
        self.num_classes = 40
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
        return ['/scratch/ys6310/graphprompter/dataset/ogbn-arxiv/processed_data.pt']

    def get_idx_split(self):
        np_filename = f'/scratch/ys6310/graphprompter/dataset/split/semi_ogbn-arxiv.npy'
        loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
        # Convert the numpy arrays or non-Python int types to standard Python lists of int
        train_ids = [int(i) for i in loaded_data_dict['train']]

        val_ids = [int(i) for i in loaded_data_dict['val']]

        sup_np_filename = f'/scratch/ys6310/graphprompter/dataset/split/sup_ogbn-arxiv.npy'
        loaded_data_dict = np.load(sup_np_filename, allow_pickle=True).item()
        test_ids = [int(i) for i in loaded_data_dict['test']]

        print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
        
        return {'train': train_ids, 'test': test_ids, 'val': val_ids}


if __name__ == '__main__':
    dataset = ArxivDataset()

    print(dataset.graph)
    print(dataset.prompt)
    print(json.dumps(dataset[0], indent=4))

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
