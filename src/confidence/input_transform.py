import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import MinCovDet
from typing import Union, Literal, Optional, Tuple
#TODO check the math beding this agian

import torch
import torch.nn as nn
from torch_pca import PCA  #

class InputTransform(nn.Module):
    """
    Input transform for data standardization and whitening.
    """
    def __init__(
            self,
            standardize: bool = False,
            whiten: bool = False,
            robust_cov: bool = False,
            eps: float = 1e-8
    ):
        super().__init__()
        self.standardize = standardize or robust_cov
        self.whiten = whiten
        self.robust_cov = robust_cov
        self.eps = eps

        # register buffers so they move with .to(device)
        self.register_buffer('mean', None)
        self.register_buffer('std', None)
        self.register_buffer('cov_mean', None)
        self.register_buffer('whitening_matrix', None)

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        """
        Fit standardization and whitening parameters from data.
        
        Args:
            data: Input data array or tensor
            y: Unused label argument
        
        Returns:
            None
        """
        if isinstance(data, torch.Tensor):
            orig_shape = data.shape
            if data.dim() >= 2:
                data2d = data.reshape(-1, data.shape[-1])
            else:
                data2d = data.view(-1, 1)
            if self.standardize:
                m = data2d.mean(dim=0)
                s = data2d.std(dim=0)
                s = torch.where(s < self.eps, torch.ones_like(s), s)
                self.mean, self.std = m, s
                data2d = (data2d - m) / s
            if self.whiten:
                arr = data2d.cpu().numpy()
                if self.robust_cov:
                    mcd = MinCovDet().fit(arr)
                    cov = torch.from_numpy(mcd.covariance_).to(data2d.dtype)
                    cm = torch.from_numpy(mcd.location_).to(data2d.dtype)
                else:
                    cm = data2d.mean(dim=0)
                    cov = torch.from_numpy(np.cov(arr, rowvar=False)).to(data2d.dtype)
                eigv, eigvec = torch.linalg.eigh(cov)
                inv_sqrt = eigv.clamp(min=self.eps).rsqrt()
                W = eigvec @ torch.diag(inv_sqrt) @ eigvec.T
                self.cov_mean, self.whitening_matrix = cm, W
            return
        else:
            orig_shape = data.shape
            if data.ndim >= 2:
                data2d = data.reshape(-1, data.shape[-1])
            else:
                data2d = data.reshape(-1, 1)
            if self.standardize:
                m = data2d.mean(axis=0)
                s = data2d.std(axis=0)
                s[s < self.eps] = 1.0
                data2d = (data2d - m) / s
                self.mean = torch.from_numpy(m.astype(np.float32))
                self.std = torch.from_numpy(s.astype(np.float32))
            if self.whiten:
                if self.robust_cov:
                    mcd = MinCovDet().fit(data2d)
                    cov = mcd.covariance_
                    cm = mcd.location_
                else:
                    cm = data2d.mean(axis=0)
                    cov = np.cov(data2d, rowvar=False)
                eigv, eigvec = np.linalg.eigh(cov)
                inv_sqrt = 1.0 / np.sqrt(np.maximum(eigv, self.eps))
                W = eigvec @ np.diag(inv_sqrt) @ eigvec.T
                self.cov_mean = torch.from_numpy(cm.astype(np.float32))
                self.whitening_matrix = torch.from_numpy(W.astype(np.float32))
            return

    def transform(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply fitted transform to data.
        
        Args:
            data: Input data array or tensor
            y: Unused label argument
        
        Returns:
            Transformed data
        """
        if isinstance(data, torch.Tensor):
            x = data
            if self.standardize and self.mean is not None:
                x = (x - self.mean) / self.std
            if self.whiten and self.whitening_matrix is not None:
                x = (x - self.cov_mean) @ self.whitening_matrix
            return x
        else:
            x = data
            if self.standardize and self.mean is not None:
                m = self.mean.numpy()
                s = self.std.numpy()
                x = (x - m) / s
            if self.whiten and self.whitening_matrix is not None:
                cm = self.cov_mean.numpy()
                W = self.whitening_matrix.numpy()
                x = (x - cm) @ W
            return x

    def forward(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        """Forward pass calling transform."""
        return self.transform(data)


class InputTransformImage(nn.Module):
    """
    Input transforms that downsamples the image and optionally applies random projection.

    """
    def __init__(self, reduce_dims=(2, 2),reshape_image_shape=None,average_pool=True,
                 rp_dim: Optional[int] = None, rp_method: str = "gaussian", rp_seed: Optional[int] = None,
                 rp_normalize_rows: bool = True):
        super().__init__()
        self.reduce_dims = reduce_dims
        self.avg_pool = nn.AdaptiveAvgPool2d(reduce_dims) if average_pool else nn.AdaptiveMaxPool2d(reduce_dims)
        self.reshape_image_shape = reshape_image_shape #potentially reshape to an image if input is already flat

        self.rp = None
        self.rp_dim = rp_dim
        # store constructor metadata so we can reconstruct on load
        self.rp_method = rp_method
        self.rp_seed = rp_seed
        self.rp_normalize_rows = rp_normalize_rows
        self._rp_skipped = False  # whether RP was intentionally skipped because pooled dim <= rp_dim
        if rp_dim is not None:
            # create RP submodule
            self.rp = RandomProjectionModule(
                n_components=rp_dim,
                method=rp_method,
                seed=rp_seed,
                normalize_rows=rp_normalize_rows,
            )

    def transform(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply pooling and random projection to image data.
        
        Args:
            data: Input image data
            y: Unused label argument
        
        Returns:
            Transformed features
        """
        was_tensor = torch.is_tensor(data)
        # reshape image if requested
        if self.reshape_image_shape is not None:
            if was_tensor:
                data = data.view(-1, *self.reshape_image_shape)
            else:
                data = data.reshape(-1, *self.reshape_image_shape)

        # convert numpy -> torch for pooling
        if not was_tensor:
            data_t = torch.from_numpy(data.astype(np.float32))
        else:
            data_t = data

        pooled = self.avg_pool(data_t).flatten(start_dim=1)  # [B, F]
        pooled_dim = int(pooled.shape[1])

        # apply random projection only if pooled_dim > requested rp_dim
        if self.rp is not None and pooled_dim > int(self.rp.n_components):
            if was_tensor:
                out = self.rp.transform(pooled.float())
                return out
            else:
                out = self.rp.transform(pooled.detach().cpu().numpy())
                return out
        else:
            # RP not configured or pooled dim too small -> skip RP
            return pooled if was_tensor else pooled.detach().cpu().numpy()

    def forward(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        """Forward pass calling transform."""
        return self.transform(data)

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        """
        Fit pooling and random projection parameters.
        
        Args:
            data: Input image data
            y: Unused label argument
        
        Returns:
            Self for chaining
        """
        if self.reshape_image_shape is not None:
            if isinstance(data, torch.Tensor):
                data = data.view(-1, *self.reshape_image_shape)
            else:
                data = data.reshape(-1, *self.reshape_image_shape)

        if isinstance(data, torch.Tensor):
            data_t = data
        else:
            data_t = torch.from_numpy(data.astype(np.float32))

        pooled = self.avg_pool(data_t).flatten(start_dim=1)  # [N, F]
        pooled_dim = int(pooled.shape[1])

        if self.rp is not None:
            if pooled_dim > int(self.rp.n_components):
                self.rp.fit(pooled)
                self._rp_skipped = False
            else:
                self._rp_skipped = True
        return self

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        """
        if any(k.startswith("rp.") for k in state_dict.keys()) and self.rp is not None:
            self.rp.load_state_dict({k[3:]: v for k, v in state_dict.items() if k.startswith("rp.")}, strict=False)

        res = super().load_state_dict(state_dict, strict=False)
        # update skip flag if rp exists but not fitted
        if self.rp is not None:
            self._rp_skipped = not getattr(self.rp, "_fitted", False)
        return res


class PCAInputModule(nn.Module):
    """
    PCA-based dimensionality reduction using sklearn or PyTorch.
    """
    def __init__(self, n_components: int = 2, keep_first: bool = True, backend: str = "sklearn"):
        super().__init__()
        self.n_components = n_components
        self.keep_first = keep_first
        self.backend = backend
        # always start with an empty buffer so state_dict() works
        self.register_buffer("proj_matrix", torch.empty(0))
        self._fitted = False

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None, device=None) -> "PCAInputModule":
        """
        Fit PCA projection matrix.
        
        Args:
            data: Input data for fitting
            y: Unused label argument
            device: Target device for projection matrix
        
        Returns:
            Self for chaining
        """
        # Flatten everything except batch -> [B, F]
        if torch.is_tensor(data):
            x = data.detach()
            if x.dim() == 0:
                raise ValueError("PCAInputModule.fit: scalar input not supported")
            if x.dim() == 1:
                arr2d = x.view(1, -1)
            else:
                arr2d = x.contiguous().view(x.shape[0], -1)
            arr = arr2d.cpu().numpy() if self.backend == "sklearn" else arr2d
        else:
            x = data
            if x.ndim == 0:
                raise ValueError("PCAInputModule.fit: scalar input not supported")
            if x.ndim == 1:
                arr = x.reshape(1, -1)
            else:
                arr = x.reshape(x.shape[0], -1)

        n_features = arr.shape[1]

        if self.backend == "sklearn":
            from sklearn.decomposition import PCA
            pca = PCA(
                n_components=self.n_components if self.keep_first else None,
                svd_solver="auto"
            )
            pca.fit(arr)
            comps = pca.components_
        else:
            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(arr, full_matrices=False)
            comps = Vh.cpu().numpy()

        # Select components
        if self.keep_first:
            selected = comps[: self.n_components]
        else:
            selected = comps[self.n_components: self.n_components * 2]

        device = device or (data.device if torch.is_tensor(data) else "cpu")
        pm = torch.tensor(selected, dtype=torch.float32, device=device)

        self.proj_matrix.resize_as_(pm).copy_(pm)
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA projection to input.
        
        Args:
            x: Input tensor
        
        Returns:
            Projected features
        """
        # Flatten everything except batch; output [B, C] or [C] for 1D input
        if self.proj_matrix.numel() == 0:
            raise RuntimeError("PCAInputModule is not fitted. Call fit() first.")
        if x.dim() == 0:
            raise ValueError("PCAInputModule.forward: scalar input not supported")
        if x.dim() == 1:
            x2d = x.view(1, -1)
            D_in = x2d.shape[1]
            if D_in != int(self.proj_matrix.shape[1]):
                raise ValueError(f"PCAInputModule.forward: input features {D_in} != fitted {int(self.proj_matrix.shape[1])}")
            out2d = x2d @ self.proj_matrix.T
            return out2d.view(-1)
        else:
            B = x.shape[0]
            x2d = x.contiguous().view(B, -1)
            D_in = x2d.shape[1]
            if D_in != int(self.proj_matrix.shape[1]):
                raise ValueError(f"PCAInputModule.forward: input features {D_in} != fitted {int(self.proj_matrix.shape[1])}")
            out2d = x2d @ self.proj_matrix.T
            return out2d

    def transform(self, data: Union[np.ndarray, torch.Tensor], y=None):
        """
        Transform data using fitted PCA.
        
        Args:
            data: Input data array or tensor
            y: Unused label argument
        
        Returns:
            Transformed data
        """
        if torch.is_tensor(data):
            return self.forward(data.float())
        else:
            x = torch.from_numpy(data.astype(np.float32))
            out = self.forward(x).detach().cpu().numpy()
            return out

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load projection matrix from checkpoint."""
        if "proj_matrix" in state_dict:
            ckpt_pm = state_dict["proj_matrix"]
            if self.proj_matrix.shape != ckpt_pm.shape:
                # re-register the buffer with correct shape
                self.register_buffer("proj_matrix", ckpt_pm.clone())
                # remove from state_dict so super() doesn't try again
                del state_dict["proj_matrix"]

        super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._fitted = self.proj_matrix.numel() > 0




class PCAInputModuleTorch(nn.Module):
    """
    PCA dimensionality reduction using torch_pca backend.
    """
    def __init__(self, n_components: int = 2, whiten=False):
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.pca: Optional[PCA] = None
        self._fitted = False

    def fit(self, data: torch.Tensor, y=None) -> "PCAInputModuleTorch":
        """
        Fit PCA model on input data.
        
        Args:
            data: Input tensor for fitting
            y: Unused label argument
        
        Returns:
            Self for chaining
        """
        if data.dim() == 1:
            data = data.unsqueeze(0)
        else:
            data = data.view(data.shape[0], -1)

        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca.fit(data)
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA projection.
        
        Args:
            x: Input tensor
        
        Returns:
            Projected features
        """
        if not self._fitted:
            raise RuntimeError("PCAInputModuleTorch is not fitted. Call fit() first.")
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.shape[0], -1)
        return self.pca.transform(x)

    def state_dict(self, *args, **kwargs):
        """Serialize PCA parameters to state dict."""
        sd = super().state_dict(*args, **kwargs)
        if self._fitted and self.pca is not None:
            # flatten PCA params into the dict
            sd["pca.components_"] = self.pca.components_
            sd["pca.mean_"] = self.pca.mean_
            sd["pca.explained_variance_"] = self.pca.explained_variance_
            sd["pca.explained_variance_ratio_"] = self.pca.explained_variance_ratio_
            sd["pca.singular_values_"] = self.pca.singular_values_
            sd["pca.noise_variance_"] = self.pca.noise_variance_
            sd["pca.n_components_"] = torch.tensor(self.pca.n_components_)
            sd["pca.n_samples_"] = torch.tensor(self.pca.n_samples_)
            sd["pca.n_features_in_"] = torch.tensor(self.pca.n_features_in_)
            sd["pca.whiten"] = torch.tensor(int(self.pca.whiten))
        return sd

    def load_state_dict(self, state_dict, strict=True):
        """Restore PCA parameters from state dict."""
        # Extract PCA params
        pca_keys = [k for k in state_dict if k.startswith("pca.")]
        if pca_keys:
            self.pca = PCA(
                n_components=int(state_dict["pca.n_components_"].item()),
                whiten=bool(state_dict["pca.whiten"].item()),
            )
            self.pca.components_ = state_dict["pca.components_"]
            self.pca.mean_ = state_dict["pca.mean_"]
            self.pca.explained_variance_ = state_dict["pca.explained_variance_"]
            self.pca.explained_variance_ratio_ = state_dict["pca.explained_variance_ratio_"]
            self.pca.singular_values_ = state_dict["pca.singular_values_"]
            self.pca.noise_variance_ = state_dict["pca.noise_variance_"]
            self.pca.n_components_ = int(state_dict["pca.n_components_"].item())
            self.pca.n_samples_ = int(state_dict["pca.n_samples_"].item())
            self.pca.n_features_in_ = int(state_dict["pca.n_features_in_"].item())
            self._fitted = True
            for k in pca_keys:
                state_dict.pop(k)

        super().load_state_dict(state_dict, strict)

    def to(self, *args, **kwargs):
        """Move PCA model to device."""
        super().to( *args, **kwargs)
        if self.pca is not None:
            self.pca.to( *args, **kwargs)
        return self



class TokenPooling(nn.Module):
    """
    Pool token representations to single vector per sample.
    
    Methods:
        - 'mean': average patch tokens (exclude CLS)
        - 'max': max over patch tokens (exclude CLS)
        - 'cls': CLS token only
        - 'cls+mean': concatenate CLS + mean of patches
        - 'cls+max': concatenate CLS + max of patches
        - integer string: subsample every nth token
    """

    def __init__(self, method: str = "mean", attn_hidden: Optional[int] = None):
        super().__init__()
        self.method = method
        self.attn_hidden = attn_hidden

    def forward(self, x: torch.Tensor):
        """
        Pool tokens from transformer output.
        
        Args:
            x: Token tensor [B, L, D] or [L, D]
        
        Returns:
            Pooled features
        """
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)

        cls_token = x[:, 0:1, :]  # CLS token
        patch_tokens = x[:, 1:, :]  # All patch tokens

        if self.method == "mean":
            out = patch_tokens.mean(dim=1)
        elif self.method == "max":
            out, _ = patch_tokens.max(dim=1)
        elif self.method == "cls":
            out = cls_token.squeeze(1)
        elif self.method == "cls+mean":
            patch_mean = patch_tokens.mean(dim=1)
            out = torch.cat([cls_token.squeeze(1), patch_mean], dim=1)
        elif self.method == "cls+max":
            patch_max, _ = patch_tokens.max(dim=1)
            out = torch.cat([cls_token.squeeze(1), patch_max], dim=1)
        elif self.method.isdigit():
            n = int(self.method)
            out = patch_tokens[:, ::n, :]
            out = out.flatten(1)  # flatten token dim if needed
        else:
            raise ValueError(f"Unknown pooling method {self.method}")

        return out if batched else out.squeeze(0)


class L2Normalization(nn.Module):
    """L2 normalization layer."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply L2 normalization.
        
        Args:
            x: Input tensor
        
        Returns:
            L2-normalized tensor
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + self.eps)


class RandomProjectionModule(nn.Module):
    """
    Random linear projection using dense Gaussian or orthogonal matrices.
    """
    def __init__(
        self,
        n_components: int,
        method: Literal["gaussian", "orthogonal"] = "gaussian",
        seed: int = None,
        normalize_rows: bool = True,
    ):
        super().__init__()
        self.n_components = n_components
        self.method = method.lower()
        self.seed = seed
        self.normalize_rows = normalize_rows
        self.proj = None
        self._fitted = False

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None) -> "RandomProjectionModule":
        """
        Generate and store random projection matrix.
        
        Args:
            data: Input data (shape inferred from second dimension)
            y: Unused label argument
        
        Returns:
            Self for chaining
        """
        if isinstance(data, torch.Tensor):
            x = data
            if x.dim() == 0:
                raise ValueError("RandomProjectionModule.fit: scalar input not supported")
            if x.dim() == 1:
                data2d = x.view(1, -1)
            else:
                data2d = x.contiguous().view(x.shape[0], -1)
            n_features = int(data2d.shape[1])
            device = data2d.device
            dtype = data2d.dtype
        else:
            x = data.reshape(data.shape[0], -1) if data.ndim > 1 else data.reshape(1, -1)
            n_features = int(x.shape[1])
            device = torch.device("cpu")
            dtype = torch.float32

        if self.seed is not None:
            old_rand_state = torch.get_rng_state()
            torch.manual_seed(self.seed)

        if self.method == "gaussian":
            W = torch.randn(self.n_components, n_features, device=device, dtype=dtype)
            W /= math.sqrt(self.n_components)
        elif self.method == "orthogonal":
            M = torch.randn(max(self.n_components, n_features), n_features, device=device, dtype=dtype)
            Q, _ = torch.linalg.qr(M, mode='reduced')
            W = Q[:self.n_components] * math.sqrt(n_features / self.n_components)
        else:
            raise ValueError(f"Unknown method '{self.method}'")

        if self.normalize_rows:
            norms = W.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            W = W / norms

        self.proj = nn.Linear(n_features, self.n_components, bias=False).to(device=device, dtype=dtype)
        self.proj.weight.data = W
        self.proj.weight.requires_grad = False
        self._fitted = True
        if self.seed is not None:
            torch.set_rng_state(old_rand_state)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random projection.
        
        Args:
            x: Input tensor
        
        Returns:
            Projected features
        """
        # Flatten everything except batch; output [B, C] or [C] for 1D input
        if self.proj is None:
            raise RuntimeError("RandomProjectionModule not fitted. Call fit() first.")
        if x.dim() == 0:
            raise ValueError("RandomProjectionModule.forward: scalar input not supported")
        if x.dim() == 1:
            x2d = x.view(1, -1)
            D_in = x2d.shape[1]
            if D_in != int(self.proj.weight.shape[1]):
                raise ValueError(f"RandomProjectionModule.forward: input features {D_in} != fitted {int(self.proj.weight.shape[1])}")
            out2d = self.proj(x2d)
            return out2d.view(-1)
        else:
            B = x.shape[0]
            x2d = x.contiguous().view(B, -1)
            D_in = x2d.shape[1]
            if D_in != int(self.proj.weight.shape[1]):
                raise ValueError(f"RandomProjectionModule.forward: input features {D_in} != fitted {int(self.proj.weight.shape[1])}")
            out2d = self.proj(x2d)
            return out2d

    def transform(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform data using random projection.
        
        Args:
            data: Input array or tensor
            y: Unused label argument
        
        Returns:
            Transformed data
        """
        if isinstance(data, torch.Tensor):
            return self.forward(data.float())
        else:
            x = torch.from_numpy(data.astype(np.float32))
            out = self.forward(x).detach().cpu().numpy()
            return out

    def load_state_dict(self, state_dict, strict: bool = True):
        """Restore projection from checkpoint."""
        # state_dict may be Tensors (e.g., from safetensors)
        w = state_dict.get("proj.weight", None)
        if w is not None:
            # Ensure plain Tensor
            if isinstance(w, torch.nn.Parameter):
                w = w.data
            n_out, n_in = int(w.shape[0]), int(w.shape[1])
            # Create proj if missing or mismatched
            if self.proj is None or self.proj.weight.shape != w.shape:
                self.proj = nn.Linear(n_in, n_out, bias=False)
                self.proj.weight.requires_grad = False
            # Keep metadata consistent
            self.n_components = n_out
        result = super().load_state_dict(state_dict, strict=False)  # be permissive
        self._fitted = self.proj is not None
        return result


import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from typing import Union


class SparseRandomProjectionModule(nn.Module):
    """
    Memory-efficient random projection using sparse CSR matrices.
    """

    def __init__(
        self,
        n_components: int,
        density: Union[float, str] = 'auto',
        seed: int = None
    ):
        super().__init__()
        self.n_components = n_components
        self.density = density
        self.seed = seed

        # Initialize as None so state_dict can load real tensors without shape mismatch
        self.register_buffer('sparse_weight_values', torch.tensor([], dtype=torch.float32), persistent=True)
        self.register_buffer('sparse_weight_indices', torch.empty(2, 0, dtype=torch.long), persistent=True)
        self.register_buffer('sparse_weight_shape', torch.empty(2, dtype=torch.long), persistent=True)

        self._fitted = False

    def _make_sparse_random_matrix(
        self,
        n_components: int,
        n_features: int,
        density: float,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Generate sparse random projection matrix.
        
        Args:
            n_components: Number of output dimensions
            n_features: Number of input features
            density: Sparsity level (fraction of non-zeros)
            dtype: Data type for values
            device: Target device
        
        Returns:
            Tuple of (indices, values, shape)
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Expected number of non-zeros
        nnz = int(density * n_components * n_features)
        s = 1.0 / density

        # Random positions
        row_indices = np.random.randint(0, n_components, size=nnz)
        col_indices = np.random.randint(0, n_features, size=nnz)

        # Random values ±sqrt(s)/sqrt(n_components)
        values = np.random.choice(
            [-np.sqrt(s) / np.sqrt(n_components), np.sqrt(s) / np.sqrt(n_components)],
            size=nnz
        )

        # Build sparse matrix
        components = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_components, n_features),
            dtype=np.float32
        )
        components.eliminate_zeros()

        coo = components.tocoo()

        indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
        values = torch.from_numpy(coo.data.copy()).float()

        return indices.to(device), values.to(device), torch.Size([n_components, n_features])

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None) -> "SparseRandomProjectionModule":
        """
        Generate sparse random projection matrix from data shape.
        
        Args:
            data: Input data for shape inference
            y: Unused label argument
        
        Returns:
            Self for chaining
        """
        if isinstance(data, torch.Tensor):
            x = data.view(data.shape[0], -1) if data.dim() > 1 else data.view(1, -1)
            n_features = x.shape[1]
            device = x.device
            dtype = x.dtype
        else:
            x = data.reshape(data.shape[0], -1) if data.ndim > 1 else data.reshape(1, -1)
            n_features = x.shape[1]
            device = torch.device("cpu")
            dtype = torch.float32

        # Auto density from sklearn: 1 / sqrt(n_features)
        if self.density == 'auto':
            density = min(1.0, 1.0 / np.sqrt(n_features))
        else:
            density = float(self.density)

        indices, values, shape = self._make_sparse_random_matrix(
            self.n_components, n_features, density, dtype, device
        )

        # Update buffers
        self.sparse_weight_indices = indices
        self.sparse_weight_values = values
        self.sparse_weight_shape = torch.tensor(list(shape), dtype=torch.long, device=device)

        self._fitted = True
        return self

    def _get_sparse_weight(self, device=None):
        """
        Retrieve sparse weight tensor on target device.
        
        Args:
            device: Target device
        
        Returns:
            Sparse COO tensor
        """
        if not self._fitted or self.sparse_weight_indices is None:
            raise RuntimeError("SparseRandomProjectionModule not fitted.")
        indices = self.sparse_weight_indices.to(device)
        values = self.sparse_weight_values.to(device)
        shape = tuple(self.sparse_weight_shape.tolist())
        return torch.sparse_coo_tensor(indices, values, size=shape).coalesce()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sparse random projection.
        
        Args:
            x: Input tensor
        
        Returns:
            Projected features
        """
        if not self._fitted:
            raise RuntimeError("SparseRandomProjectionModule not fitted. Call fit() first.")
        if x.dim() == 0:
            raise ValueError("SparseRandomProjectionModule.forward: scalar input not supported")

        x_2d = x.view(-1, int(self.sparse_weight_shape[1]))
        sparse_weight = self._get_sparse_weight(device=x.device)
        return torch.sparse.mm(sparse_weight, x_2d.t()).t()

    def transform(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform data using sparse projection.
        
        Args:
            data: Input array or tensor
            y: Unused label argument
        
        Returns:
            Transformed data
        """
        if isinstance(data, torch.Tensor):
            return self.forward(data.float())
        else:
            x = torch.from_numpy(data.astype(np.float32))
            return self.forward(x).detach().cpu().numpy()

    def load_state_dict(self, state_dict, strict: bool = True):
        """Restore sparse projection buffers from checkpoint."""
        # Replace buffers directly with checkpoint values
        for name in ["sparse_weight_values", "sparse_weight_indices", "sparse_weight_shape"]:
            if name in state_dict:
                tensor = state_dict[name]
                self.register_buffer(name, tensor)
        result = super().load_state_dict(state_dict, strict=False)
        # Mark as fitted if buffers are present
        if (self.sparse_weight_indices is not None
                and self.sparse_weight_values is not None
                and self.sparse_weight_shape is not None):
            self._fitted = True
        return result



class InputTransformCollapse(nn.Module):
    """
    Collapse the spatial H,W dimensions to a single point per channel.
    - No constructor parameters.
    - For input shapes:
        - [B, C, H, W] -> returns [B, C] (global average over H,W)
        - [C, H, W] -> returns [C]
        - [B, F] or [F] -> returned unchanged
    - fit() is a no-op to match reducer API.
    """
    def __init__(self):
        super().__init__()

    def fit(self, data: Union[np.ndarray, torch.Tensor], y=None) -> "InputTransformCollapse":
        """
        Fit method (no-op for collapse).
        
        Args:
            data: Unused
            y: Unused
        
        Returns:
            Self for chaining
        """
        # No fitting required for collapse reducer
        return self

    def transform(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        """
        Collapse spatial dimensions via global average.
        
        Args:
            data: Input array or tensor
            y: Unused label argument
        
        Returns:
            Collapsed features
        """
        if isinstance(data, torch.Tensor):
            x = data
            if x.dim() == 4:
                # [B, C, H, W] -> [B, C]
                return x.mean(dim=(2, 3))
            elif x.dim() == 3:
                # [C, H, W] -> [C]
                return x.mean(dim=(1, 2))
            else:
                # already collapsed or vector; return as-is
                return x
        else:
            x = data
            if x.ndim == 4:
                # numpy [B, C, H, W]
                return x.mean(axis=(2, 3))
            elif x.ndim == 3:
                # numpy [C, H, W]
                return x.mean(axis=(1, 2))
            else:
                return x

    def forward(self, data: Union[np.ndarray, torch.Tensor], y=None) -> Union[np.ndarray, torch.Tensor]:
        """Forward pass calling transform."""
        return self.transform(data, y)


FEATURE_REDUCER_REGISTRY = {
    "pca": PCAInputModule,
    "rp": RandomProjectionModule,
    "sparse_rp": SparseRandomProjectionModule,
    "token_pool": TokenPooling,
    "image_transform": InputTransformImage,
    "collapse_image": InputTransformCollapse,   # << added key for simple collapse reducer
    "pca_torch": PCAInputModuleTorch,
}

def create_feature_reducer(name: str, **kwargs) -> nn.Module:
    """
    Create a feature reducer by name. The reducer must provide:
      - fit(data: 2D torch.Tensor) -> self
      - forward(x: 2D torch.Tensor) -> 2D torch.Tensor
    """
    if name is None:
        return None
    #check if name is a torch nn module class
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name(**kwargs)

    key = name.lower()
    if key not in FEATURE_REDUCER_REGISTRY:
        raise ValueError(f"Unknown feature reducer '{name}'. Available: {list(FEATURE_REDUCER_REGISTRY)}")
    cls = FEATURE_REDUCER_REGISTRY[key]
    return cls(**kwargs)

