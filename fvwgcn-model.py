

import argparse
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


def load_interactions(path: str) -> Tuple[int, int, List[Tuple[int, int]]]:
    """Load user–item interaction pairs from a text file.

    Each line in the file should contain two integers separated by whitespace:
    ``user_id item_id``. User and item indices are assumed to be zero‑based.

    Returns
    -------
    num_users : int
        The maximum user index plus one.
    num_items : int
        The maximum item index plus one.
    interactions : List[Tuple[int, int]]
        A list of (user_id, item_id) pairs.
    """
    users = []
    items = []
    interactions: List[Tuple[int, int]] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u_str, i_str = line.split()
            u = int(u_str)
            i = int(i_str)
            users.append(u)
            items.append(i)
            interactions.append((u, i))
    num_users = max(users) + 1 if users else 0
    num_items = max(items) + 1 if items else 0
    return num_users, num_items, interactions


def load_sparse_adj(path: str) -> torch.sparse.FloatTensor:
    """Load a bipartite adjacency matrix from a .npz file.

    The file should contain a SciPy sparse matrix saved with
    ``scipy.sparse.save_npz``. The matrix is expected to be of shape
    (num_users + num_items, num_users + num_items) and contain the
    bipartite graph in the upper right and lower left blocks. User–user
    and item–item blocks can be zero.

    Returns a PyTorch sparse tensor in COO format.
    """
    mat = sp.load_npz(path)
    mat = mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((mat.row, mat.col)).astype(np.int64)
    )
    values = torch.from_numpy(mat.data)
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


class Dataset:
    """A simple dataset class for implicit feedback recommender training.

    Parameters
    ----------
    interactions : List[Tuple[int, int]]
        List of (user, item) interactions.
    num_users : int
        Total number of users.
    num_items : int
        Total number of items.
    seed : int, optional
        Random seed for splitting.
    val_ratio : float
        Fraction of interactions per user to assign to the validation set.
    test_ratio : float
        Fraction of interactions per user to assign to the test set.
    """

    def __init__(
        self,
        interactions: List[Tuple[int, int]],
        num_users: int,
        num_items: int,
        seed: int = 42,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ) -> None:
        self.num_users = num_users
        self.num_items = num_items
        self.interactions = interactions
        # Build user‑wise interaction lists
        user_items: List[List[int]] = [list() for _ in range(num_users)]
        for u, i in interactions:
            user_items[u].append(i)
        # Split per user
        self.train_pairs: List[Tuple[int, int]] = []
        self.val_pairs: List[Tuple[int, int]] = []
        self.test_pairs: List[Tuple[int, int]] = []
        rng = random.Random(seed)
        for u in range(num_users):
            items = user_items[u]
            if not items:
                continue
            rng.shuffle(items)
            n_total = len(items)
            n_test = int(n_total * test_ratio)
            n_val = int(n_total * val_ratio)
            test_items = items[:n_test]
            val_items = items[n_test : n_test + n_val]
            train_items = items[n_test + n_val :]
            self.train_pairs.extend((u, i) for i in train_items)
            self.val_pairs.extend((u, i) for i in val_items)
            self.test_pairs.extend((u, i) for i in test_items)
        # For negative sampling, build a set of positive items per user
        self.user_pos = [set(lst) for lst in user_items]

    def sample_pair(self) -> Tuple[int, int, int]:
        """Sample a positive and a negative item for a random user.

        Returns a tuple (user, positive_item, negative_item).
        """
        # Choose a random user with at least one interaction
        while True:
            u = random.randint(0, self.num_users - 1)
            pos_items = list(self.user_pos[u])
            if pos_items:
                break
        pos_item = random.choice(pos_items)
        # Sample a negative item
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_pos[u]:
                break
        return u, pos_item, neg_item


class TuningGCN(nn.Module):
    """Tuning GCN that generates localized attention weight matrices (LAWM).

    This module repeatedly samples local subgraphs around randomly selected
    users, performs a single graph convolution on each sampled subgraph,
    applies a learnable linear projection to the resulting features, and
    aggregates the outer products of these projected features to form a
    localized attention weight matrix. Averaging across multiple sampled
    subgraphs yields a stable estimate of the dynamic weight matrix.

    Parameters
    ----------
    num_users : int
        Number of users.
    num_items : int
        Number of items.
    embed_dim : int
        Dimensionality of the user/item embeddings.
    sample_users : int
        Number of users to sample in each subgraph.
    sample_times : int
        Number of subgraphs to sample per forward pass.
    user_pos_dict : List[set]
        Mapping from user to a set of positively interacted items. Used
        to construct user–item adjacency for subgraphs.
    device : torch.device
        Device on which computations will be performed.
    threshold : float, optional
        Cosine similarity threshold for user–user edges. Only similarities
        above this value will be considered when constructing the user–user
        block of a subgraph.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int,
        sample_users: int,
        sample_times: int,
        user_pos_dict: List[set],
        device: torch.device,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.sample_users = sample_users
        self.sample_times = sample_times
        self.user_pos_dict = user_pos_dict
        self.device = device
        self.threshold = threshold
        # Learnable projection matrix used to transform features before
        # computing the outer product. Parameterized as a linear layer.
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a localized attention weight matrix.

        Parameters
        ----------
        user_emb : torch.Tensor
            Tensor of shape (num_users, embed_dim) containing user
            embeddings.
        item_emb : torch.Tensor
            Tensor of shape (num_items, embed_dim) containing item
            embeddings.

        Returns
        -------
        torch.Tensor
            A tensor of shape (embed_dim, embed_dim) representing the
            averaged localized attention weight matrix.
        """
        W_sum = torch.zeros(
            self.embed_dim, self.embed_dim, device=self.device, dtype=user_emb.dtype
        )
        eps = 1e-8
        # Precompute norms for cosine similarity
        user_norms = F.normalize(user_emb, p=2, dim=1)
        for _ in range(self.sample_times):
            # Sample a subset of users uniformly at random
            sampled_users = random.sample(range(self.num_users), self.sample_users)
            # Collect items interacted by sampled users
            sampled_items_set = set()
            for u in sampled_users:
                sampled_items_set.update(self.user_pos_dict[u])
            sampled_items = list(sampled_items_set)
            # Build mapping from global indices to subgraph indices
            user_to_idx = {u: i for i, u in enumerate(sampled_users)}
            item_to_idx = {i: j + len(sampled_users) for j, i in enumerate(sampled_items)}
            subgraph_size = len(sampled_users) + len(sampled_items)
            # Construct adjacency for the sampled subgraph
            # We store indices and values for a COO sparse matrix
            rows: List[int] = []
            cols: List[int] = []
            vals: List[float] = []
            # User–user edges based on cosine similarity threshold
            for i, u in enumerate(sampled_users):
                # compute similarity with other users in the sample
                sim_vec = torch.matmul(user_norms[u], user_norms[sampled_users].t())
                for j, sim in enumerate(sim_vec):
                    if sim > self.threshold and i != j:
                        rows.append(i)
                        cols.append(j)
                        vals.append(1.0)
            # User–item edges from interactions
            for u in sampled_users:
                for itm in self.user_pos_dict[u]:
                    if itm in item_to_idx:
                        rows.append(user_to_idx[u])
                        cols.append(item_to_idx[itm])
                        vals.append(1.0)
                        # Symmetric edge for bipartite graph
                        rows.append(item_to_idx[itm])
                        cols.append(user_to_idx[u])
                        vals.append(1.0)
            # Create sparse adjacency matrix
            if rows:
                indices = torch.tensor([rows, cols], device=self.device)
                values = torch.tensor(vals, device=self.device)
                sub_A = torch.sparse.FloatTensor(indices, values, (subgraph_size, subgraph_size))
            else:
                # Empty adjacency; skip this sample
                continue
            # Construct initial feature matrix: concatenate user and item embeddings
            sub_feat = torch.zeros(
                subgraph_size, self.embed_dim, device=self.device, dtype=user_emb.dtype
            )
            for u in sampled_users:
                idx = user_to_idx[u]
                sub_feat[idx] = user_emb[u]
            for itm in sampled_items:
                idx = item_to_idx[itm]
                sub_feat[idx] = item_emb[itm]
            # Perform a single GCN layer: H^(1) = A * X
            H1 = torch.sparse.mm(sub_A, sub_feat)
            # Project features
            H_proj = self.projection(H1)
            # Compute outer product and accumulate
            W_t = H_proj.t() @ H_proj  # shape (d, d)
            # Normalize matrix to avoid scale explosion
            norm = torch.norm(W_t) + eps
            W_sum += W_t / norm
        # Average across samples
        if self.sample_times > 0:
            W_avg = W_sum / float(self.sample_times)
        else:
            W_avg = W_sum
        return W_avg


class TunedGCN(nn.Module):
    """Tuned GCN that performs propagation with static and adaptive weights.

    The Tuned GCN takes a normalized adjacency matrix and a dynamic weight
    matrix generated by the Tuning GCN. It maintains a set of static
    weight matrices for each GCN layer and combines them multiplicatively
    with the adaptive matrix at each layer. The output embeddings are a
    weighted sum of the layer outputs.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the graph (users + items).
    embed_dim : int
        Dimensionality of the embeddings.
    num_layers : int
        Number of graph convolution layers.
    adjacency : torch.sparse.FloatTensor
        Symmetric normalized adjacency matrix of shape (num_nodes, num_nodes).
    device : torch.device
        Device for computation.
    """

    def __init__(
        self,
        num_nodes: int,
        embed_dim: int,
        num_layers: int,
        adjacency: torch.sparse.FloatTensor,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.adj = adjacency.coalesce().to(device)
        self.device = device
        # Static weight matrices for each layer
        self.static_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(embed_dim, device=device)) for _ in range(num_layers)]
        )
        # Trainable coefficients for layer combination
        self.alpha = nn.Parameter(torch.ones(num_layers + 1, device=device))

    def forward(
        self,
        all_emb: torch.Tensor,
        W_lawm: torch.Tensor,
    ) -> torch.Tensor:
        """Perform graph convolutions with adaptive weights.

        Parameters
        ----------
        all_emb : torch.Tensor
            Initial embeddings of shape (num_nodes, embed_dim) for users and items.
        W_lawm : torch.Tensor
            Localized adaptive weight matrix of shape (embed_dim, embed_dim).

        Returns
        -------
        torch.Tensor
            Final embeddings of shape (num_nodes, embed_dim).
        """
        # Normalize adaptive matrix
        W_adapt = W_lawm
        # Collect embeddings at each layer (including initial)
        layer_embs = [all_emb]
        h = all_emb
        for l in range(self.num_layers):
            # Graph convolution: normalized adjacency times features
            h = torch.sparse.mm(self.adj, h)
            # Apply combined weight
            W_comb = self.static_weights[l] @ W_adapt
            h = h @ W_comb
            layer_embs.append(h)
        # Stack embeddings and combine with learned coefficients
        layer_embs = torch.stack(layer_embs, dim=0)  # shape (L+1, N, d)
        alpha = F.softmax(self.alpha, dim=0)  # normalized combination weights
        # Weighted sum along layers
        final_emb = torch.einsum('l,lnd->nd', alpha, layer_embs)
        return final_emb


class DualGCN(nn.Module):
    """Wrapper model that orchestrates the tuning and tuned GCN modules."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int,
        num_layers: int,
        adjacency: torch.sparse.FloatTensor,
        user_pos_dict: List[set],
        sample_users: int,
        sample_times: int,
        device: torch.device,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.device = device
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self.tuning = TuningGCN(
            num_users=num_users,
            num_items=num_items,
            embed_dim=embed_dim,
            sample_users=sample_users,
            sample_times=sample_times,
            user_pos_dict=user_pos_dict,
            device=device,
            threshold=threshold,
        )
        self.tuned = TunedGCN(
            num_nodes=num_users + num_items,
            embed_dim=embed_dim,
            num_layers=num_layers,
            adjacency=adjacency,
            device=device,
        )

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute user and item embeddings after dual GCN propagation."""
        # Generate adaptive weight matrix using current embeddings
        user_emb = self.user_embeddings.weight
        item_emb = self.item_embeddings.weight
        W_lawm = self.tuning(user_emb, item_emb)
        # Concatenate embeddings for all nodes
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        all_emb = self.tuned(all_emb, W_lawm)
        # Split back into user and item embeddings
        out_user_emb = all_emb[: self.num_users]
        out_item_emb = all_emb[self.num_users :]
        return out_user_emb, out_item_emb


def bpr_loss(
    user_emb: torch.Tensor,
    pos_item_emb: torch.Tensor,
    neg_item_emb: torch.Tensor,
    l2_reg: float = 1e-4,
) -> torch.Tensor:
    """Compute the Bayesian Personalized Ranking (BPR) loss.

    Parameters
    ----------
    user_emb : torch.Tensor
        Embeddings for a batch of users, shape (batch_size, embed_dim).
    pos_item_emb : torch.Tensor
        Embeddings for the corresponding positive items, shape (batch_size, embed_dim).
    neg_item_emb : torch.Tensor
        Embeddings for the corresponding negative items, shape (batch_size, embed_dim).
    l2_reg : float
        L2 regularization weight applied to embeddings.

    Returns
    -------
    torch.Tensor
        Scalar BPR loss.
    """
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    # L2 regularization on user and item embeddings
    loss += l2_reg * (user_emb.norm(p=2) ** 2 + pos_item_emb.norm(p=2) ** 2 + neg_item_emb.norm(p=2) ** 2)
    return loss


def normalize_adj(adj: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
    """Symmetrically normalize a sparse adjacency matrix.

    Given a sparse adjacency matrix A (without self‑loops), this function
    computes D^{-1/2} A D^{-1/2}, where D is the diagonal degree matrix.
    Self‑loops can be added externally if desired.
    """
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    row, col = indices
    # Compute degrees
    deg = torch.zeros(adj.size(0), device=values.device)
    deg.scatter_add_(0, row, values)
    # Avoid division by zero
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    # Normalize values
    norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(indices, norm_values, adj.size())


def train(args: argparse.Namespace) -> None:
    """Main training routine."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load interactions
    num_users, num_items, interactions = load_interactions(args.data_path)
    # Initialize dataset and build splits
    dataset = Dataset(
        interactions=interactions,
        num_users=num_users,
        num_items=num_items,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    # Load adjacency matrix
    adj = load_sparse_adj(args.adj_path)
    # Add self-loops and normalize
    num_nodes = num_users + num_items
    # Create identity matrix for self-loops
    eye_indices = torch.arange(num_nodes, device=device)
    eye_indices = torch.stack([eye_indices, eye_indices], dim=0)
    eye_values = torch.ones(num_nodes, device=device)
    eye = torch.sparse_coo_tensor(eye_indices, eye_values, (num_nodes, num_nodes))
    adj = (adj + eye).coalesce()
    adj_norm = normalize_adj(adj).to(device)
    # Build model
    model = DualGCN(
        num_users=num_users,
        num_items=num_items,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        adjacency=adj_norm,
        user_pos_dict=dataset.user_pos,
        sample_users=args.sample_users,
        sample_times=args.sample_times,
        device=device,
        threshold=args.threshold,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        # Sample mini-batches
        for _ in range(args.samples_per_epoch):
            u, i, j = dataset.sample_pair()
            optimizer.zero_grad()
            # Forward pass to get updated embeddings
            out_user_emb, out_item_emb = model()
            # Extract embeddings for sampled indices
            user_vec = out_user_emb[u].unsqueeze(0)
            pos_vec = out_item_emb[i].unsqueeze(0)
            neg_vec = out_item_emb[j].unsqueeze(0)
            # Compute BPR loss
            loss = bpr_loss(user_vec, pos_vec, neg_vec, l2_reg=args.l2_reg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / float(args.samples_per_epoch)
        print(f"Epoch {epoch:03d}, Loss = {avg_loss:.6f}")
    print("Training complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DualGCN Recommender Training")
    parser.add_argument('--data_path', type=str, required=True, help='Path to interactions file (u i per line)')
    parser.add_argument('--adj_path', type=str, required=True, help='Path to preprocessed adjacency (.npz)')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GCN layers in tuned module')
    parser.add_argument('--sample_users', type=int, default=100, help='Number of users sampled per LAWM computation')
    parser.add_argument('--sample_times', type=int, default=3, help='Number of subgraph samples to average for LAWM')
    parser.add_argument('--threshold', type=float, default=0.5, help='Cosine similarity threshold for user-user edges in tuning GCN')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for Adam optimizer')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='L2 regularization weight in BPR loss')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--samples_per_epoch', type=int, default=1000, help='Number of BPR samples per epoch')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio for dataset splitting')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test ratio for dataset splitting')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)