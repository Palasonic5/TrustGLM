import torch
from torch_geometric.utils import mask_to_index, index_to_mask


def batch_subgraph(edge_index,
                   node_ids,
                   num_nodes,
                   num_hops=3,
                   fans_out=(50, 50, 50)):

    subset_list, edge_index_sub_list, mapping_list, batch_list = [], [], [], []

    row, col = edge_index
    inc_num = 0
    batch_id = 0

    for node_idx in node_ids:
        subsets = [node_idx.flatten()]
        node_mask = row.new_empty(num_nodes, dtype=torch.bool)

        for _ in range(num_hops):
            node_mask.fill_(False)

            node_mask[subsets[-1]] = True
            edge_mask = torch.index_select(node_mask, 0, row)

            neighbors = col[edge_mask]
            if len(neighbors) > fans_out[_]:
                perm = torch.randperm(len(neighbors))[:fans_out[_]]
                neighbors = neighbors[perm]

            subsets.append(neighbors)

        subset, ind = torch.unique(torch.cat(subsets), return_inverse=True)

        node_mask = index_to_mask(subset, size=num_nodes)
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index_sub = edge_index[:, edge_mask]

        # Relabel Node
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long, device=edge_index.device)
        node_idx[subset] = torch.arange(node_mask.sum().item(), device=edge_index.device)
        edge_index_sub = node_idx[edge_index_sub]

        # Batching Graph
        edge_index_sub += inc_num

        subset_list.append(subset)
        edge_index_sub_list.append(edge_index_sub)
        mapping_list.append(inc_num + ind[0].item())
        batch_list.extend([batch_id for _ in range(len(subset))])

        inc_num += len(subset)
        batch_id += 1

    subset = torch.cat(subset_list)
    mapping = torch.as_tensor(mapping_list)
    batch = torch.as_tensor(batch_list)
    edge_index_sub = torch.cat(edge_index_sub_list, dim=1)

    return subset, edge_index_sub, mapping, batch

def batch_subgraph_new(edge_indices_dict, node_ids, num_nodes, num_hops=3, fans_out=(50, 50, 50)):
    """
    Batch subgraph generation for a list of node IDs, each with its unique edge_index.

    Parameters:
        edge_indices_dict (dict): A dictionary where keys are node IDs and values are their specific edge_index tensors.
        node_ids (Tensor): The node IDs for which subgraphs are extracted.
        num_nodes (int): Total number of nodes in the graph.
        num_hops (int): Number of hops to consider for subgraph extraction.
        fans_out (tuple): Maximum number of neighbors sampled at each hop.

    Returns:
        tuple: subset (nodes in the subgraph), edge_index_sub (edges in the subgraph),
               mapping (mapping of nodes), batch (batch IDs for nodes).
    """
    subset_list, edge_index_sub_list, mapping_list, batch_list = [], [], [], []

    inc_num = 0  # 用于累积节点编号偏移量
    batch_id = 0

    for node_idx in node_ids:
        node_idx = node_idx.item()
        # 从字典中获取当前节点的 edge_index
        edge_index = edge_indices_dict[node_idx]
        row, col = edge_index

        # 初始化子集，从当前节点开始
        subsets = [torch.tensor([node_idx], dtype=torch.long)]
        node_mask = row.new_empty(num_nodes, dtype=torch.bool)

        # 遍历 num_hops，逐层扩展邻居节点
        for hop in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            edge_mask = torch.index_select(node_mask, 0, row)

            neighbors = col[edge_mask]
            if len(neighbors) > fans_out[hop]:
                perm = torch.randperm(len(neighbors))[:fans_out[hop]]
                neighbors = neighbors[perm]

            subsets.append(neighbors)

        # 合并所有节点，并确保唯一性
        subset, ind = torch.unique(torch.cat(subsets), return_inverse=True)

        # 根据子集生成子图
        node_mask = index_to_mask(subset, size=num_nodes)
        edge_mask = node_mask[row] & node_mask[col]
        edge_index_sub = edge_index[:, edge_mask]

        # 重新编号子图中的节点
        node_idx_map = torch.zeros(node_mask.size(0), dtype=torch.long, device=edge_index.device)
        node_idx_map[subset] = torch.arange(node_mask.sum().item(), device=edge_index.device)
        edge_index_sub = node_idx_map[edge_index_sub]

        # 为批次中的子图调整边索引编号
        edge_index_sub += inc_num

        # 更新结果列表
        subset_list.append(subset)
        edge_index_sub_list.append(edge_index_sub)
        mapping_list.append(inc_num + ind[0].item())
        batch_list.extend([batch_id] * len(subset))

        inc_num += len(subset)  # 更新累计编号
        batch_id += 1  # 更新批次 ID

    # 合并所有结果
    subset = torch.cat(subset_list)
    mapping = torch.tensor(mapping_list)
    batch = torch.tensor(batch_list)
    edge_index_sub = torch.cat(edge_index_sub_list, dim=1)

    return subset, edge_index_sub, mapping, batch



class TAGCollator(object):
    def __init__(self, graph):
        self.graph = graph

    def __call__(self, original_batch):
        mybatch = {}
        for k in original_batch[0].keys():
            mybatch[k] = [d[k] for d in original_batch]

        subset, edge_index_sub, mapping, batch = batch_subgraph(
            edge_index=self.graph.edge_index,
            node_ids=torch.tensor(mybatch['id']),
            num_nodes=self.graph.num_nodes
        )

        mybatch['x'] = self.graph.x[subset]
        mybatch['y'] = self.graph.y[subset]
        mybatch['edge_index'] = edge_index_sub
        mybatch['mapping'] = mapping
        mybatch['batch'] = batch

        return mybatch

class TAGCollator_new(object):
    def __init__(self, graph, edge_indices_dict):
        self.graph = graph
        self.edge_indices_dict = edge_indices_dict
    def __call__(self, original_batch):
        mybatch = {}
        for k in original_batch[0].keys():
            mybatch[k] = [d[k] for d in original_batch]

        # subset, edge_index_sub, mapping, batch = batch_subgraph(
        #     edge_index=self.graph.edge_index,
        #     node_ids=torch.tensor(mybatch['id']),
        #     num_nodes=self.graph.num_nodes
        # )
        subset, edge_index_sub, mapping, batch = batch_subgraph_new(
        edge_indices_dict=self.edge_indices_dict,
        node_ids=torch.tensor(mybatch['id']),
        num_nodes=self.graph.num_nodes
        )

        mybatch['x'] = self.graph.x[subset]
        mybatch['y'] = self.graph.y[subset]
        mybatch['edge_index'] = edge_index_sub
        mybatch['mapping'] = mapping
        mybatch['batch'] = batch

        return mybatch


collate_funcs = {
    'cora_sup': TAGCollator,
    'cora_sup_nettack': TAGCollator_new,
    'cora_sup_metattack': TAGCollator,
    'cora_sup_shuffle': TAGCollator,
    'cora_sup_ln50': TAGCollator,
    'cora_sup_ln100': TAGCollator,
    'cora_sup_st': TAGCollator,
    'cora_sup_prbcd_g': TAGCollator,
    'cora_sup_prbcd_l': TAGCollator_new,
    'cora_sup_lnc50': TAGCollator,
    'cora_sup_lnc100': TAGCollator,
    'cora_sup_lnt': TAGCollator,
    'cora_sup_clnt': TAGCollator,
    'citeseer': TAGCollator,
    'pubmed_sup': TAGCollator,
    'pubmed_sup_nettack': TAGCollator_new,
    'pubmed_sup_metattack': TAGCollator,
    'pubmed_sup_shuffle': TAGCollator,
    'pubmed_sup_ln50': TAGCollator,
    'pubmed_sup_ln100': TAGCollator,
    'pubmed_sup_prbcd_g': TAGCollator,
    'pubmed_sup_st': TAGCollator,
    'pubmed_sup_prbcd_l': TAGCollator_new,
    'pubmed_sup_lnc50': TAGCollator,
    'pubmed_sup_lnc100': TAGCollator,
    'pubmed_sup_lnt': TAGCollator,
    'pubmed_sup_clnt': TAGCollator,
    'arxiv_sup': TAGCollator,
    'arxiv_sup_shuffle': TAGCollator,
    'arxiv_sup_prbcd_g':TAGCollator,
    'arxiv_sup_prbcd_l': TAGCollator_new,
    'products_sup': TAGCollator,
    'products_sup_nettack': TAGCollator_new,
    "products_sup_shuffle": TAGCollator,
    'products_sup_ln50': TAGCollator,
    'products_sup_ln100': TAGCollator,
    'products_sup_prbcd_g': TAGCollator,
    'products_sup_st': TAGCollator,
    'products_sup_prbcd_l': TAGCollator_new,
    'products_sup_lnc50': TAGCollator,
    'products_sup_lnc100': TAGCollator,
    'products_sup_lnt': TAGCollator,
    'products_sup_clnt': TAGCollator,
    'cora_semi': TAGCollator,
    'cora_semi_nettack': TAGCollator_new,
    'cora_semi_metattack': TAGCollator,
    'pubmed_semi': TAGCollator,
    'pubmed_semi_nettack': TAGCollator_new,
    'arxiv_semi': TAGCollator,
    'products_semi': TAGCollator,
    'products_semi_nettack': TAGCollator_new,
    "sports_semi": TAGCollator,
    "sports_sup": TAGCollator,
    "sports_sup_shuffle": TAGCollator,
    'sports_sup_ln50': TAGCollator,
    'sports_sup_ln100': TAGCollator,
    'sports_sup_st': TAGCollator,
    'sports_sup_lnc50': TAGCollator,
    'sports_sup_lnc100': TAGCollator,
    'sports_sup_lnt': TAGCollator,
    'sports_sup_clnt': TAGCollator,
    "computers_semi": TAGCollator,
    "computers_sup": TAGCollator,
    "computers_sup_shuffle": TAGCollator,
    'computers_sup_ln50': TAGCollator,
    'computers_sup_ln100': TAGCollator,
    'computers_sup_st': TAGCollator,
    'computers_sup_lnc50': TAGCollator,
    'computers_sup_lnc100': TAGCollator,
    'computers_sup_lnt': TAGCollator,
    'computers_sup_clnt': TAGCollator,
    "photo_semi": TAGCollator,
    "photo_sup": TAGCollator
}
