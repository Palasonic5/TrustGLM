import argparse
import os
import pickle
import numpy as np
import scipy.sparse as sp
from nettack import utils, GCN
from nettack import nettack as ntk

def main():
    parser = argparse.ArgumentParser(description='Nettack Attack Script')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv_sup',
                        help='Name of the dataset (default: ogbn-arxiv_sup)')
    parser.add_argument('--attr_type', type=str, default='sbert',
                        help='Attribute type (default: sbert for LLaGA, bert for GraphTranslator, GIA for GraphPrompter, etc.)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (default: 0)')
    args = parser.parse_args()

    dataset = args.dataset
    attr_type = args.attr_type
    gpu_id = args.gpu_id

    # Load data
    print(f"Loading data for dataset: {dataset}, attr_type: {attr_type}")
    _A_obs, _X_obs, _z_obs = utils.load_pt(
        f'dataset/{dataset}/processed_data.pt',
        dataset,
        attr_type
    )

    # Symmetrize adjacency
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1

    _X_obs = _X_obs.astype('float32')

    _N = _A_obs.shape[0]
    _K = _z_obs.max() + 1

    _Z_obs = np.eye(_K)[_z_obs].reshape(-1, _K)

    # Preprocess adjacency
    _An = utils.preprocess_graph(_A_obs)
    sizes = [16, _K]
    degrees = _A_obs.sum(0).A1

    # Load train/val/test split
    np_filename = f'/dataset/split/{dataset}.npy'
    loaded_data_dict = np.load(np_filename, allow_pickle=True).item()
    split_train = [int(i) for i in loaded_data_dict['train']]
    split_val = [int(i) for i in loaded_data_dict['val']]
    split_unlabeled = [int(i) for i in loaded_data_dict['test']]

    # Create directory for attack output
    save_path = f"attack/Graph_attack/nettack/output/{dataset}_{attr_type}"
    os.makedirs(save_path, exist_ok=True)

    # Train the surrogate model only once
    print("Training surrogate model...")
    surrogate_model = GCN.GCN(sizes, _An, _X_obs, with_relu=False, name="surrogate", gpu_id=gpu_id)
    surrogate_model.train(split_train, split_val, _Z_obs)

    # Extract weights after training
    W1 = surrogate_model.W1.eval(session=surrogate_model.session)
    W2 = surrogate_model.W2.eval(session=surrogate_model.session)
    print("Surrogate model trained and weights extracted.")

    # Save and reload weights (optional, if you want to verify saving/reloading)
    weights_dir = f"attack/Graph_attack/nettack/output/surrogate_weights/{dataset}_{attr_type}"
    os.makedirs(weights_dir, exist_ok=True)  # ensure directory exists

    with open(os.path.join(weights_dir, "surrogate_weights.pkl"), "wb") as f:
        pickle.dump({"W1": W1, "W2": W2}, f)
    print(f"Weights saved to {weights_dir}/surrogate_weights.pkl")

    with open(os.path.join(weights_dir, "surrogate_weights.pkl"), "rb") as f:
        weights = pickle.load(f)

    W1 = weights["W1"]
    W2 = weights["W2"]
    print("Weights loaded successfully.")

    # Iterate through test nodes
    for u in split_unlabeled:
        print(f"Processing node: {u}")

        # Initialize Nettack for the current node
        nettack = ntk.Nettack(_A_obs, _X_obs, _z_obs, W1, W2, u, verbose=True)

        # Set parameters for the attack
        direct_attack = True
        n_influencers = 1 if direct_attack else 5
        n_perturbations = int(degrees[u])  # Number of perturbations
        perturb_features = False
        perturb_structure = True

        # Perform the attack
        nettack.reset()
        nettack.attack_surrogate(
            n_perturbations,
            perturb_structure=perturb_structure,
            perturb_features=perturb_features,
            direct=direct_attack,
            n_influencers=n_influencers
        )

        # Get modified adjacency matrix
        modified_adjacency = nettack.adj_preprocessed

        # Save the modified adjacency matrix for the current node
        out_file_prefix = f"{save_path}/modified_adjacency_node_{u}"
        if sp.issparse(modified_adjacency):
            # Save as .npz for sparse matrix
            sp.save_npz(f"{out_file_prefix}.npz", modified_adjacency)
            print(f"Modified adjacency matrix for node {u} saved as {out_file_prefix}.npz")
        else:
            # Save as .npy for dense matrix
            np.save(f"{out_file_prefix}.npy", modified_adjacency)
            print(f"Modified adjacency matrix for node {u} saved as {out_file_prefix}.npy")

        # Mark as processed (optional)
        # already.add(u)

    print("All nodes processed.")

if __name__ == "__main__":
    main()
