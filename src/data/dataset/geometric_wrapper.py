from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch_scatter import scatter_mean, scatter_max


class GeometricLabelDatasetWrapper(torch.utils.data.Dataset):
    """
    A wrapper for a datasets from the torch_geometric library. These return a torch_geometric.data.Data object.
    This class wraps them so that the the tuple (dataobject, label) is returned. For use with model.classifier.py.
    """

    def __init__(self, dataset):
        """
        Initialize the GeometricDatasetWrapper.
        Args:
            dataset (Dataset): The dataset to wrap.
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        Args:
            idx: The index of the item to retrieve.

        Returns:
            #TODO how to format tuple return in google docstring
        """
        data = self.dataset[idx]
        label = data.y.squeeze()
        return data, label


from torch.utils.data import Dataset


class GeometricsDatasetWrapper(Dataset):
    """
    Wrapper that wraps around a torch_geometric Dataset and returns the position and label to use with normal dataloaders.
    This only works when the underlying dataset returns a Data object with a pos attribute that have
    the same number of points.
    """

    def __init__(self, dataset):
        self.dataset = dataset  # dataset is expected to be a list or another Dataset of torch_geometric.data.Data objects

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data.pos, data.y.squeeze()  # returns only the position and label (can be modified depending on model input)


import torch
from torch.utils.data import Dataset


class TensorGeometricsDatasetWrapper(Dataset):
    """
    Wrapper that wraps around a torch_geometric Dataset and returns the position and label to use with normal dataloaders.
    This only works when the underlying dataset returns a Data object with a pos attribute that have
    the same number of points.
    """

    def __init__(self, dataset):
        self.dataset = dataset  # dataset is expected to be a list or another Dataset of torch_geometric.data.Data objects

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, y = self.dataset[idx]
        return data.pos, y  # returns only the position and label (can be modified depending on model input)


class TensorGeometricModelWrapper(torch.nn.Module):
    """
    Wrapper for a torch_geometric model that receives a tuple of (pos, y) as input and creates
    a Data object from it that is passed to the model.
    """

    def __init__(self, model):
        super(TensorGeometricModelWrapper, self).__init__()
        self.model = model  # model should be a torch_geometric.nn GNN model

    def forward(self, pos):
        # pos and y are batched tensors from a DataLoader
        # need to reconstruct the original Data objects for torch_geometric models

        # reconstruct Data objects assuming uniform sizes (optional: batch from individual Data objects instead)
        data_list = [Data(pos=p) for p in pos]
        batch = Batch.from_data_list(data_list)

        return self.model(batch)


class BatchNormalizeScale(torch.nn.Module):
    """
    Batch normalization and scaling of the input data to [-1, 1].
    """

    def __init__(self):
        super().__init__()

    def single_data(self, data: Data) -> Data:
        data = self.center(data)

        assert data.pos is not None
        scale = (1.0 / data.pos.abs().max()) * 0.999999
        data.pos = data.pos * scale

        return data

    def batch_data(self, data: Batch) -> Batch:
        pos = data.pos
        batch_idx = data.batch

        # Centering
        mean = scatter_mean(pos, batch_idx, dim=0)
        pos = pos - mean[batch_idx]

        # Scaling
        abs = pos.abs()
        max_values, _ = scatter_max(abs, batch_idx, dim=0)
        max_per_batch = max_values.max(dim=1, keepdim=True)[0]

        scale = (1.0 / (max_per_batch + 1e-8)) * 0.999999
        data.pos = pos * scale[batch_idx]

        return data

    def forward(self, data):
        # Handle both single Data and batched Batch objects
        if isinstance(data, Batch):
            return self.batch_data(data)
        return self.single_data(data)


class BatchNormalizeScaleEuclidean(torch.nn.Module):
    """
    Batch normalization and scaling of the input data to [-1, 1] using Euclidean distance.
    Instead of normalizing based on the maximum absolute value along any axis,
    this variant normalizes based on the maximum Euclidean distance from the center.
    Should be rotation equivariant.
    """

    def __init__(self):
        super().__init__()

    def single_data(self, data: Data) -> Data:
        # Center the data
        assert data.pos is not None
        mean = data.pos.mean(dim=0, keepdim=True)
        data.pos = data.pos - mean

        # Calculate Euclidean distances from center
        distances = torch.norm(data.pos, dim=1)
        max_distance = distances.max()

        # Scale based on maximum Euclidean distance
        scale = (1.0 / (max_distance + 1e-8)) * 0.999999
        data.pos = data.pos * scale

        return data

    def batch_data(self, data: Batch) -> Batch:
        pos = data.pos
        batch_idx = data.batch

        # Centering
        mean = scatter_mean(pos, batch_idx, dim=0)
        pos = pos - mean[batch_idx]

        # Calculate Euclidean distances from center for each point
        distances = torch.norm(pos, dim=1, keepdim=True)

        # Find maximum distance per batch
        max_distances, _ = scatter_max(distances, batch_idx, dim=0)

        # Scale based on maximum Euclidean distance
        scale = (1.0 / (max_distances + 1e-8)) * 0.999999
        data.pos = pos * scale[batch_idx]

        return data

    def forward(self, data):
        # Handle both single Data and batched Batch objects
        if isinstance(data, Batch):
            return self.batch_data(data)
        return self.single_data(data)


class NormalizeRotationBatch:
    """Batch version that matches PyG's NormalizeRotation"""

    def __init__(self, max_points: int = -1, sort: bool = False, ensure_proper_rotation: bool = False):
        self.max_points = max_points
        self.sort = sort
        self.ensure_proper_rotation = ensure_proper_rotation

    def forward(self, batch: Batch) -> Batch:
        pos, batch_idx = batch.pos, batch.batch
        new_pos = torch.zeros_like(pos)
        has_normals = ('normal' in batch) and (batch.normal is not None)
        if has_normals:
            new_normals = torch.zeros_like(batch.normal)

        for b in batch_idx.unique(sorted=True):
            mask = (batch_idx == b)
            p_full = pos[mask]

            # Subsample for eigenvector estimation
            p = p_full
            if self.max_points > 0 and p.size(0) > self.max_points:
                perm = torch.randperm(p.size(0), device=p.device)
                p = p[perm[:self.max_points]]

            # Center and covariance
            p_centered = p - p.mean(dim=0, keepdim=True)
            C = p_centered.t() @ p_centered
            e, v = torch.linalg.eig(C)
            v = v.real

            # Optional sorting
            if self.sort:
                e_real = torch.view_as_real(e)[:, 0]
                indices = e_real.argsort(descending=True)
                v = v.t()[indices].t()

            # Ensure proper rotation
            if self.ensure_proper_rotation:
                if torch.det(v) < 0:
                    v[:, -1] *= -1

            # Rotate full cloud
            new_pos[mask] = p_full @ v

            # Rotate normals if present
            if has_normals:
                n_full = batch.normal[mask]
                new_normals[mask] = F.normalize(n_full @ v, dim=-1)

        batch.pos = new_pos
        if has_normals:
            batch.normal = new_normals
        return batch


class NormalizeRotationVectorized:
    """
    Vectorized batch version of PCA-based orientation normalization.

    Aligns each point cloud in a batch to its principal axes. Includes
    options for deterministic eigenvector signing, ensuring the resulting
    transformation is a proper rotation (determinant = +1), and optional
    random sign augmentation during training.
    """

    def __init__(self,
                 sort: bool = False,
                 ensure_proper_rotation: bool = True,
                 fix_sign: bool = False):
        """
        Args:
            max_points (int): The maximum number of points to use for PCA
                from each cloud. If -1, uses all points. Defaults to -1.
            sort (bool): If True, sorts eigenvectors by their corresponding
                eigenvalues in descending order. Defaults to False.
            ensure_proper_rotation (bool): If True, ensures the final
                transformation matrix is a proper rotation (determinant = +1),
                preventing reflections. Defaults to True.
            fix_sign (bool): If True, flips eigenvectors so they align
                consistently with the centroid of each point cloud.
                Defaults to False.
        """
        self.sort = sort
        self.ensure_proper_rotation = ensure_proper_rotation
        self.fix_sign = fix_sign

    @torch.no_grad()
    def forward(self, batch: Batch, randomize=False, use_svd_for_rotation: bool = False) -> Batch:
        # Preserve original dtype to optionally convert back at the end
        pos_orig = batch.pos
        orig_dtype = pos_orig.dtype
        device = pos_orig.device
        # Force float64 for all internal computation
        dtype = torch.float64
        pos = pos_orig.to(device=device, dtype=dtype)
        batch_idx = batch.batch.to(device=device)

        N = pos.size(0)
        B = int(batch_idx.max().item()) + 1

        # choose all points (or subsample mask) - keep typed as float64
        use_mask = torch.ones(N, dtype=torch.bool, device=device)
        pos_sel = pos[use_mask]
        batch_sel = batch_idx[use_mask]

        # scatter sums for covariance (float64)
        ones = torch.ones((pos_sel.size(0), 1), device=device, dtype=dtype)
        count = scatter_add(ones, batch_sel, dim=0, dim_size=B)  # (B,1)
        sum_x = scatter_add(pos_sel, batch_sel, dim=0, dim_size=B)  # (B,3)
        outer = pos_sel.unsqueeze(2) * pos_sel.unsqueeze(1)  # (N,3,3)
        sum_xx = scatter_add(outer.view(-1, 9), batch_sel, dim=0, dim_size=B).view(B, 3, 3)

        mu = sum_x / count.clamp(min=1.0)
        mu_outer = mu.unsqueeze(2) @ mu.unsqueeze(1)
        C = sum_xx - count.view(B, 1, 1) * mu_outer  # covariance *count

        # Regularize covariance to avoid tiny negative eigenvalues / degeneracy
        # eps scale: small absolute + relative to trace
        trace = (torch.diagonal(C, dim1=1, dim2=2).sum(dim=1) / 3.0).clamp(min=1e-12)
        eps = (1e-12 + 1e-9 * trace).view(B, 1, 1)
        C = C + eps * torch.eye(3, device=device, dtype=dtype).unsqueeze(0)

        # eigh returns (eigenvalues, eigenvectors). eigenvalues are real and ascending order.
        e, v = torch.linalg.eigh(C)  # e: (B,3) ascending, v: (B,3,3)
        # Put eigenvectors in descending order (largest first) if desired
        if self.sort:
            idx = e.argsort(dim=-1, descending=True)
            idx_expand = idx.unsqueeze(1).expand(-1, 3, -1)
            v = v.gather(dim=2, index=idx_expand)

        # Optionally use SVD-based rotation construction instead (robust orthogonal)
        if use_svd_for_rotation:
            # SVD of symmetric matrix C: U S Vh, for symmetric U==V. We use U @ Vh to get orthogonal R.
            U, S, Vh = torch.linalg.svd(C)  # U: (B,3,3), Vh: (B,3,3)
            R = U @ Vh
            v = R

        # randomize or fix_sign (ensure dtype float64)
        if randomize:
            signs = (torch.randint(0, 2, (B, 3), device=device) * 2 - 1).to(dtype=dtype)
            v = v * signs.unsqueeze(1)
            if not self.sort:
                perm_idx = torch.stack([torch.randperm(3, device=device) for _ in range(B)])
                v = torch.stack([v[b][:, perm_idx[b]] for b in range(B)])
        elif self.fix_sign:
            # deterministic sign based on centroid
            dots = torch.einsum("bi,bij->bj", mu.to(dtype=dtype), v)  # (B,3)
            signs = torch.where(dots < 0, -1.0, 1.0).to(dtype=dtype)
            v = v * signs.unsqueeze(1)

        # Ensure proper rotation (determinant +1)
        if self.ensure_proper_rotation:
            det = torch.linalg.det(v)
            flip_mask = det < 0
            if flip_mask.any():
                v[flip_mask, :, -1] *= -1

        # Rotate full cloud: pick per-node rotation and apply
        v_per_node = v[batch_idx]
        # pos: (N,3), v_per_node: (N,3,3) -> use bmm
        new_pos = torch.bmm(pos.unsqueeze(1).to(dtype=dtype), v_per_node).squeeze(1)

        # Convert back to original dtype if required
        if orig_dtype != dtype:
            batch.pos = new_pos.to(dtype=orig_dtype)
        else:
            batch.pos = new_pos

        # normals
        if hasattr(batch, "normal") and batch.normal is not None:
            n = batch.normal.to(device=device, dtype=dtype)
            new_normals = F.normalize(torch.bmm(n.unsqueeze(1), v_per_node).squeeze(1), dim=-1)
            batch.normal = new_normals.to(dtype=orig_dtype) if orig_dtype != dtype else new_normals

        return batch


class NormalizeRotationVectorizedModule(torch.nn.Module):
    """
    Thin nn.Module wrapper around NormalizeRotationVectorized so it can be used in nn.Sequential.
    The randomization behavior follows the training/eval state of the module.
    """

    def __init__(self, sort: bool = False,
                 ensure_proper_rotation: bool = True,
                 fix_sign: bool = False,
                 randomize: bool = False):
        super().__init__()
        self._transform = NormalizeRotationVectorized(
            sort=sort,
            ensure_proper_rotation=ensure_proper_rotation,
            fix_sign=fix_sign,
        )
        self.randomize = randomize

    def forward(self, batch: Batch) -> Batch:
        return self._transform.forward(batch, randomize=self.training and self.randomize)


def random_rotation_matrix(device='cpu', dtype=torch.float32):
    """Generate a random 3D rotation matrix using axis-angle."""
    u = torch.rand(3, device=device, dtype=dtype)
    theta = u[0] * 2 * torch.pi
    phi = u[1] * 2 * torch.pi
    z = u[2] * 2 - 1
    r = torch.sqrt(1 - z ** 2)
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    axis = torch.tensor([x, y, z], device=device, dtype=dtype)
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], device=device, dtype=dtype)
    R = torch.eye(3, device=device, dtype=dtype) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K
    return R


def add_to_canonical_set(pos, canonical_set, tol=1e-5):
    """Add a point cloud to the set of canonical forms with tolerance."""
    pos_flat = pos.flatten()
    for existing in canonical_set:
        if torch.allclose(pos_flat, existing, atol=tol):
            return  # Already in the set
    canonical_set.append(pos_flat)


class TransformParamDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, target_params):
        self.base_dataset = base_dataset
        self.target_params = target_params

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        target_params = self.target_params[idx]
        return x, target_params, y


if __name__ == "__main__":
    from torch_geometric.transforms import NormalizeRotation as PyGNormalizeRotation

    num_points = 20
    num_samples = 100  # Number of times to apply train-time randomization
    num_rotations = 100  # How many arbitrary rotations for eval mode
    tol = 1e-5

    norm_module = NormalizeRotationVectorizedModule(randomize=False, ensure_proper_rotation=False, sort=True,
                                                    fix_sign=True)

    # Generate a single random cloud
    pos = torch.randn(num_points, 3)
    data = Data(pos=pos)
    batch = Batch.from_data_list([data])

    # -------------------
    # Train mode: randomize PCA flips/permutations
    # -------------------
    norm_module.train()
    train_set = []
    for _ in range(num_samples):
        batch_train = deepcopy(batch)
        batch_train = norm_module(batch_train)  # randomized canonicalization
        add_to_canonical_set(batch_train.pos, train_set, tol=tol)

    # -------------------
    # Eval mode: deterministic PCA, apply arbitrary rotations
    # -------------------
    norm_module.eval()
    eval_set = []
    for _ in range(num_rotations):
        R = random_rotation_matrix()
        pos_rot = batch.pos @ R
        batch_eval = Batch.from_data_list([Data(pos=pos_rot)])
        batch_eval = norm_module(batch_eval)  # deterministic canonicalization
        add_to_canonical_set(batch_eval.pos, eval_set, tol=tol)

    print(f"Train mode unique canonical forms: {len(train_set)}")
    print(f"Eval mode unique canonical forms (after rotations): {len(eval_set)}")
    print("Expected: 48 if randomize + no det, 8 if non-random + no det, 4 if non-random + det=+1")

    # Create two simple clouds
    for i in range(50):
        cloud1 = Data(pos=torch.rand(10, 3))
        cloud2 = Data(pos=torch.rand(15, 3))
        batch = Batch.from_data_list([cloud1, cloud2])

        # PyG reference
        pyg_transform = PyGNormalizeRotation(max_points=-1, sort=False)
        target_positions = torch.cat([pyg_transform(d).pos for d in batch.to_data_list()], dim=0)

        # Loop batch
        batch_loop = deepcopy(batch)
        norm_loop = NormalizeRotationBatch(max_points=-1, sort=False)
        batch_loop = norm_loop.forward(batch_loop)

        # Vectorized batch
        batch_vec = deepcopy(batch)
        norm_vec = NormalizeRotationVectorized(max_points=-1, sort=False)
        batch_vec = norm_vec.forward(batch_vec)
        # Compare to PyG reference
        # first two cant exact match due to sign ambiguity of eigenvectors
        print("Loop vs PyG max diff:", (batch_loop.pos - target_positions).abs().max())
        print("Vectorized vs PyG max diff:", (batch_vec.pos - target_positions).abs().max())

        print("Loop vs Vectorized max diff:", (batch_loop.pos - batch_vec.pos).abs().max())

        assert torch.allclose(batch_vec.pos, batch_loop.pos, atol=1e-4)

    # print the pos
    print("PyG positions:\n", target_positions)
    print("Loop positions:\n", batch_loop.pos)
    print("Vectorized positions:\n", batch_vec.pos)
