from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components
import torch

def load_pt(file_name, dataset_name, attr_type=None):
    """
    Load a SparseGraph from a PyTorch .pt file.

    Parameters
    ----------
    file_name : str
        Name of the .pt file to load.

    Returns
    -------
    adj_matrix : sp.csr_matrix
        Adjacency matrix in sparse format.
    attr_matrix : sp.csr_matrix
        Attribute matrix in sparse format.
    labels : np.ndarray
        Array of labels.
    """
    # Load the processed_data.pt file
    data = torch.load(file_name)

    # Extract edge_index and convert to adjacency matrix
    edge_index = data['edge_index'].numpy()
    num_nodes = data['x'].shape[0]
    adj_matrix = sp.csr_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes)
    )

    if attr_type == "bert":
        embedding_file = f"dataset/{dataset_name}/bert_node_embeddings.pt"
        embeddings = torch.load(embedding_file).numpy()
        if embeddings.shape[0] != num_nodes:
            raise ValueError("Number of nodes in the embeddings does not match the number of nodes in the graph.")
        attr_matrix = sp.csr_matrix(embeddings)
    elif attr_type == "sbert":
        embedding_file = f"dataset/{dataset_name}/sbert_x.pt"
        embeddings = torch.load(embedding_file).numpy()
        if embeddings.shape[0] != num_nodes:
            raise ValueError("Number of nodes in the embeddings does not match the number of nodes in the graph.")
        attr_matrix = sp.csr_matrix(embeddings)
    elif attr_type == "GIA":
        feature_path = f"dataset/{dataset_name}/GIA.emb"
        features = np.load(feature_path)
        if features.shape[0] != num_nodes:
            raise ValueError("Number of nodes in the GIA features does not match the number of nodes in the graph.")
        attr_matrix = sp.csr_matrix(features)
    else:
        attr_matrix = sp.csr_matrix(data['x'].numpy())

    labels = data['y'].numpy()

    return adj_matrix, attr_matrix, labels

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized