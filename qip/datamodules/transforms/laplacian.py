import numpy as np
import scipy
import torch

from torch_geometric.typing import Optional, OptTensor
from torch_geometric.utils import get_laplacian

from admet_prediction.datamodules.transforms.base import TransformBase
from admet_prediction.typing import Data, Optional, OptTensor


def eigvec_normalizer(eig_vecs, eig_vals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """
    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = np.linalg.norm(eig_vecs, ord=1, axis=0, keepdims=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = np.linalg.norm(eig_vecs, ord=2, axis=0, keepdims=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = np.max(np.abs(eig_vecs), axis=0)

    elif normalization == "wavelength":
        # These are placeholders that we can fill if needed.
        raise NotImplementedError(f"Unsupported normalization `{normalization}`")

    elif normalization == "wavelength-asin":
        # These are placeholders that we can fill if needed.
        raise NotImplementedError(f"Unsupported normalization `{normalization}`")

    elif normalization == "wavelength-soft":
        # These are placeholders that we can fill if needed.
        raise NotImplementedError(f"Unsupported normalization `{normalization}`")

    else:
        raise NotImplementedError(f"Unsupported normalization `{normalization}`")

    denom = np.tile(np.clip(denom, a_min=eps, a_max=None), (eig_vecs.shape[0], 1))
    eig_vecs = eig_vecs / denom

    return eig_vecs


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm="L2"):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.
    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    num_nodes = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = np.clip(evals, a_min=0, a_max=None)

    # Normalize and pad eigen vectors.
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if num_nodes < max_freqs:
        eig_vecs = np.pad(evects, ((0, 0), (0, max_freqs - num_nodes)), "constant", constant_values=np.nan)
    else:
        eig_vecs = evects

    # Pad and save eigenvalues.
    if num_nodes < max_freqs:
        eig_vals = np.pad(evals, (0, max_freqs - num_nodes), "constant", constant_values=np.nan)
    else:
        eig_vals = evals

    eig_vals = np.tile(eig_vals, (num_nodes, 1))
    eig_vals = np.expand_dims(eig_vals, axis=-1)

    return eig_vals, eig_vecs


def get_laplacian_features(
    data,
    max_freqs=3,
    eigvec_norm="L2",
    eigval_inverse=False,
    eigval_norm=False,
    remove_first=False,
    edge_weight: OptTensor = None,
    normalization=None,
):
    # Eigen values and vectors.
    evals, evects = None, None
    # Basic preprocessing of the input graph.
    num_nodes = data["num_nodes"]  # Explicitly given number of nodes
    edge_index = torch.tensor(data["edge_index"])

    # If only one node or no edges, set eigenvalue/vector to zero
    if num_nodes == 1 or edge_index.shape[1] < 1:
        # eigenvalue size (num_nodes, max_freqs, 1)
        evals = np.zeros((num_nodes, max_freqs, 1))
        # eigenvector size (num_nodes, max_freqs)
        evects = np.zeros((num_nodes, max_freqs))
    else:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        edge_list, edge_attr = get_laplacian(
            edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes, normalization=normalization
        )
        row, col = edge_list
        assert edge_attr.shape[0] == row.shape[0]
        laplacian = scipy.sparse.coo_matrix((edge_attr, (row, col)), (num_nodes, num_nodes))
        evals, evects = np.linalg.eigh(
            laplacian.astype(np.float32).toarray()
        )  # Hack to convert to float32 for linear algebra

        evals, evects = get_lap_decomp_stats(evals=evals, evects=evects, max_freqs=max_freqs, eigvec_norm=eigvec_norm)

    # If remove_first is true, remove the smallest eigen value (which is 0) and first eigen vector (which are all 1s)
    # the size then becomes [num_nodes, max_freq-1]
    if remove_first:
        evals = evals[:, 1:]
        evects = evects[:, 1:]

    # If eigval_inverse is true, take the inverse of the eigen values (apart from values close to 0, will stay the same)
    # recommend to use with the remove_first
    if eigval_inverse:
        # remove nans and fill with 0
        nan_mask = np.isnan(evals)
        evals[nan_mask] = 0.0
        evals[evals > 1e-10] = 1.0 / evals[evals > 1e-10]

    # If eigval_norm is true, normalize the eigen values, recommend to use with the eigval_inverse and remove_first
    if eigval_norm:
        # this is doing the Frobenius norm by default
        # basically sum(eval_i^2)^(1/2)
        norm = np.linalg.norm(evals[0])
        if norm == 0:
            evals = evals
        else:
            evals = evals / norm
    if remove_first:
        assert evals.shape == (num_nodes, max_freqs - 1, 1)
        assert evects.shape == (num_nodes, max_freqs - 1)
    else:
        assert evals.shape == (num_nodes, max_freqs, 1)
        assert evects.shape == (num_nodes, max_freqs)

    return laplacian.todense(), (evals, evects)


class LaplacianGenerator(TransformBase):
    """
    Get Laplacian matrix, Laplacian eigen values, Laplacian eigen vectors
    """

    def __init__(
        self,
        max_freqs=3,
        eigvec_norm="L2",
        eigval_inverse=False,
        eigval_norm=False,
        remove_first=False,
        edge_weight: OptTensor = None,
        normalization: Optional[str] = None,
    ):
        self.max_freqs = max_freqs
        self.eigvec_norm = eigvec_norm
        self.eigval_inverse = eigval_inverse
        self.eigval_norm = eigval_norm
        self.remove_first = remove_first
        self.edge_weight = edge_weight
        self.normalization = normalization

    def transform_(self, data: Data) -> Data:
        laplacian_matrix, (eig_values, eig_vectors) = get_laplacian_features(
            data,
            max_freqs=self.max_freqs,
            eigvec_norm=self.eigvec_norm,
            eigval_inverse=self.eigval_inverse,
            eigval_norm=self.eigval_norm,
            remove_first=self.remove_first,
            edge_weight=self.edge_weight,
            normalization=self.normalization,
        )
        data["laplacian_mat"] = torch.from_numpy(laplacian_matrix)
        data["lap_eig_vals"] = torch.from_numpy(eig_values)
        data["lap_eig_vecs"] = torch.from_numpy(eig_vectors)
        return data
