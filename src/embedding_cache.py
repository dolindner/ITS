import copy
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple, List, Union

import torch
from safetensors.torch import save_file, load_file
from torch import nn
from tqdm import tqdm

from confidence.input_transform import create_feature_reducer
from confidence.utils import ModelInputOutputWrapper


# Note this uses DataLayer batches to calculate identification(mainly so check if shuffle is true) disadvantage is that
# this should have extracted the samples from the batches so that the same amount of samples is checked as now this depends on the batch size of the dataloader.
# meaning that changing batch size makes the cache outdated.
class LayerEmbeddingCache:
    """
    A class that for each model and dataset stores the extracted features at specific laytes.
    Calculates hash based on the model weights and the first instances of the dataloader.

    As features can be to large it instead accepts a list of reducer and their kwargs which are
    used to reduce the features before saving them.(For each reducer the features are stored)

    For each layer and reducer combination the extracted values are saved.
    Provides helper functions to create callables that return both the models output and intermediate features.

    This is used to automatically create models that extract intermediate features. The cahce is used to
    avoid recomputation saving some compute.

    Note batch size affects fingerprint this should likely be changed.
    Honestly the whole thing needs a rewrite due to adding to many features to this especially
    the multi reducer support.(support for the same reducer with multiple settings still not that great as well)
    """

    def __init__(self,
                 model: nn.Module,
                 dataloader,
                 cache_dir: str,
                 device: Optional[torch.device] = None,
                 dataset_fp_batches: int = 2,
                 max_batches: Optional[int] = None,
                 reducer_name: Optional[Union[str, List[str]]] = None,
                 reducer_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,  # CHANGED
                 reducer_fit_batches: Union[int, List[int]] = 0,
                 reducer_dim_threshold: Union[int, List[Union[int, Tuple[int, Optional[int]]]]] = 2048,
                 # when to spply each reducer
                 class_overrides: Optional[Union[Any, List[Any]]] = None
                 ):
        """
                Initializes the LayerEmbeddingCache, sets up directories, and validates fingerprints.

                This constructor initializes cache paths, normalizes feature reducer specifications,
                and computes deterministic SHA-256 fingerprints for both the model and the dataset.
                Note that batch size affects this, which should be changed by me.

                If an existing cache metadata file is found, it validates that the current model and
                dataset exactly match the cached historical states, raising an exception on drift.

                Args:
                    model (nn.Module): The PyTorch neural network model to extract features from.
                    dataloader (DataLoader): The data loader providing batches for fingerprinting
                        and feature extraction.
                    cache_dir (str): Root directory path where metadata, models, and safetensor
                        embeddings will be stored.
                    device (torch.device, optional): Target hardware device for model execution.
                        Defaults to the device of the model's first parameter.
                    dataset_fp_batches (int, optional): Number of initial batches to read from the
                        dataloader to compute the unique dataset fingerprint. Defaults to 2.
                    max_batches (int, optional): Upper limit on the number of batches to process
                        during extraction. If None, processes the entire dataloader.
                    reducer_name (str or List[str], optional): Identifier string (or list of strings)
                        for dimensionality reduction algorithms.
                    reducer_kwargs (Dict[str, Any] or List[Dict[str, Any]], optional): Explicit configuration
                        hyperparameters mapping to the chosen reducers.
                    reducer_fit_batches (int or List[int], optional): Number of batches used to fit
                        unsupervised reducers before performing transformation. Defaults to 0.
                    reducer_dim_threshold (int, Tuple, or List, optional): Embedding size thresholds
                        dictating when specific reducers should dynamically activate. Defaults to 2048.
                Raises:
                    RuntimeError: If the dataloader produces non-deterministic batches on back-to-back
                        evaluations, or if there is a fingerprint mismatch with a pre-existing cache file.
                """
        self.model = model
        self.dataloader = dataloader
        self.cache_dir = cache_dir
        self.output_dir = os.path.join(cache_dir, "output")
        self.input_dir = os.path.join(cache_dir, "input")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.input_dir, exist_ok=True)
        self.device = device
        self.dataset_fp_batches = dataset_fp_batches
        self.max_batches = max_batches
        os.makedirs(self.cache_dir, exist_ok=True)

        self.reducer_name = reducer_name
        self.reducer_kwargs = reducer_kwargs or {}
        self.reducer_fit_batches = reducer_fit_batches
        self.reducer_dim_threshold = reducer_dim_threshold
        self.class_overrides = class_overrides
        self.transforms_dir = os.path.join(self.cache_dir, "transforms")
        os.makedirs(self.transforms_dir, exist_ok=True)
        self._reducers_norm = self._normalize_reducers()

        self.meta_path = os.path.join(self.cache_dir, "metadata.json")
        self.y_path = os.path.join(self.cache_dir, "y.safetensors")
        self.final_output_path = os.path.join(self.cache_dir, "final_output.safetensors")

        self.model_fp = LayerEmbeddingCache._model_fingerprint(self.model)
        self.dataset_fp = LayerEmbeddingCache._dataset_fingerprint(self.dataloader, self.dataset_fp_batches)
        # Recompute to ensure dataloader determinism (e.g., no random shuffling between iterations)
        dataset_fp2 = LayerEmbeddingCache._dataset_fingerprint(self.dataloader, self.dataset_fp_batches)
        if dataset_fp2 != self.dataset_fp:
            raise RuntimeError(
                "Non-deterministic dataloader detected: dataset fingerprint differs across consecutive passes.")

        print(
            "Changes to pytorch version and gpu used can change the features slighly requiring a recompute. Please ensure that you update the cache in such scenarios")
        self.metadata = self._load_metadata()
        if self.metadata is None:
            self.metadata = {
                "version": 1.2,
                "model_fingerprint": self.model_fp,
                "dataset_fingerprint": self.dataset_fp,
                "model_class": self.model.__class__.__name__,
                "created_unix": time.time(),
                "updated_unix": time.time(),
                "layer_modes": {},
                "shapes": {},
                "dtypes": {},
                "transform_defaults": {
                    "enabled": bool(self.reducer_name),
                    "name": self.reducer_name if isinstance(self.reducer_name, str) else (
                        self.reducer_name[0] if self.reducer_name else None),
                    "kwargs": self.reducer_kwargs if isinstance(self.reducer_kwargs, dict) else (
                        self.reducer_kwargs[0] if isinstance(self.reducer_kwargs,
                                                             list) and self.reducer_kwargs else {}),
                    "fit_batches": self.reducer_fit_batches if isinstance(self.reducer_fit_batches, int) else (
                        self.reducer_fit_batches[0] if self.reducer_fit_batches else 0),
                    "dim_threshold": self.reducer_dim_threshold if isinstance(self.reducer_dim_threshold, int) else (
                        self.reducer_dim_threshold[0] if isinstance(self.reducer_dim_threshold,
                                                                    list) and self.reducer_dim_threshold else 2048),
                    "reducers": self._reducers_public(self._reducers_norm)  # CHANGED
                },
                "per_key_transform": {}
            }
            self._save_metadata()

            self._dict_input_stored_layer_modes = {}
            self._dict_output_stored_layer_modes = {}
        else:
            # Fingerprint mismatch
            if self.metadata.get("model_fingerprint") != self.model_fp:
                raise RuntimeError(
                    f"Cache fingerprint mismatch in '{self.cache_dir}'. "
                    f"Stored model differs from when cache was created. Create a new cache_dir or delete existing."
                )
            if self.metadata.get("dataset_fingerprint") != self.dataset_fp:
                raise RuntimeError(
                    f"Cache fingerprint mismatch in '{self.cache_dir}'. "
                    f"Dataset differs from when cache was created. Create a new cache_dir or delete existing."
                )
            # Ensure new keys exist on older metadata
            self.metadata.setdefault("transform_defaults", {
                "enabled": False, "name": None, "kwargs": {}, "fit_batches": 0, "dim_threshold": 2048
            })
            # Keep non-destructive merge for multi-reducer defaults (JSON-safe)
            self.metadata["transform_defaults"].setdefault("reducers",
                                                           self._reducers_public(self._reducers_norm))  # CHANGED
            self.metadata.setdefault("per_key_transform", {})
            # Lift any legacy single-variant entries into multi-variant schema
            for k, rec in list(self.metadata["per_key_transform"].items()):
                if isinstance(rec, dict) and "variants" not in rec and {"class", "name", "kwargs", "path",
                                                                        "input_dim"} <= set(rec.keys()):
                    legacy = self.metadata["per_key_transform"].pop(k)
                    self.metadata["per_key_transform"][k] = {"default": "__legacy__",
                                                             "variants": {"__legacy__": legacy}}
            self._reconcile_metadata_filesystem()

    def get_available_reducers(self, layer: str, mode: str) -> List[str]:
        """
        Query available reducer variants for a specific layer and mode based on embedding dimension. (some reducers are to expensive when dimesions gets large like pca)
        Returns a list of reducer IDs that can be applied.
        """
        # 1. Get embedding dimension: prefer original (pre-reduction) dim from per_key_transform if available
        rec = self.metadata.get("per_key_transform", {}).get(f"{layer}|{mode}")
        if rec and isinstance(rec, dict) and "variants" in rec:
            # Use input_dim from the default variant (or first available)
            vid = rec.get("default")
            if vid and vid in rec["variants"]:
                input_dim = rec["variants"][vid].get("input_dim")
                if input_dim is not None:
                    sample_dim = input_dim
                else:
                    # Fallback
                    for v in rec["variants"].values():
                        if "input_dim" in v:
                            sample_dim = v["input_dim"]
                            break
                    else:
                        # No input_dim found, fall back to sample computation
                        sample_dim = None
            else:
                sample_dim = None
        else:
            sample_dim = None

        if sample_dim is None:
            # Fallback: use shape from metadata or compute from sample
            shape_info = self.metadata.get("shapes", {}).get(layer, {}).get(mode)
            if shape_info:
                sample_dim = int(torch.tensor(shape_info[1:]).prod().item())
            else:
                # Dimension not in metadata, compute from a single sample
                wrapper = ModelInputOutputWrapper(self.model, target_layer_names=[layer], capture_modes=[mode],
                                                  flatten=False, concat=False)
                device = self.device or next(self.model.parameters()).device
                self.model.eval()
                with torch.no_grad():
                    batch = next(iter(self.dataloader))
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch.get("x")
                    x = x.to(device)
                    inter_list, _ = wrapper(x)
                    t = inter_list[0]
                    sample_dim = int(torch.tensor(t.shape[1:]).prod().item()) if t.dim() > 1 else 1

        # 2. Find applicable reducers based on dimension
        applicable_specs = self._applicable_reducers(sample_dim)
        return [spec['id'] for spec in applicable_specs]

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        if not os.path.isfile(self.meta_path):
            return None
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta
        except Exception:
            return None

    def _save_metadata(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, sort_keys=True)

    def _verify_cached_embeddings_single_batch(self, layers: List[str], modes: List[str],
                                               reducer_select: Optional[str] = None,
                                               rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        """
        Verify that newly computed embeddings match stored ones using only the first batch.
        Returns True if all match; False and warns if any differ.
        Faster than full verification; useful for detecting model/data drift.
        """
        device = self.device or next(self.model.parameters()).device
        self.model.eval()

        all_match = True
        with torch.no_grad():
            wrapper = ModelInputOutputWrapper(
                self.model,
                target_layer_names=layers,
                capture_modes=modes,
                flatten=False,
                concat=False
            )

            # Only process first batch
            batch = next(iter(self.dataloader))

            if isinstance(batch, dict):
                x = batch.get("x")
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(device)
            inter_list, _ = wrapper(x)

            # Compare with stored embeddings
            for i, (layer, mode) in enumerate(zip(layers, modes)):
                try:
                    stored = self._load_embeddings_from_disk(layer, mode, reducer_select)
                    # Compare just first batch
                    batch_size = x.size(0)
                    computed_batch = inter_list[i].cpu()
                    stored_batch = stored[:batch_size]

                    if not torch.allclose(computed_batch, stored_batch, rtol=rtol, atol=atol):
                        print(f"WARNING: Embeddings mismatch detected for layer='{layer}', mode='{mode}'")
                        max_diff = (computed_batch - stored_batch).abs().max().item()
                        print(f"First batch: max difference = {max_diff:.2e}")
                        all_match = False
                except FileNotFoundError:
                    pass  # File doesn't exist yet, skip

        if all_match:
            print("First batch embeddings verified and match stored values.")
        return all_match

    def _load_embeddings_from_disk(self, layer: str, mode: str, reducer_select: Optional[str] = None) -> torch.Tensor:
        """
        Load a stored embedding tensor  for a given layer and mode.
        If reducer variants exist, use reducer_select or default variant; otherwise load original untransformed embeddings.
        """
        # If variants exist, prefer them
        rec = self.metadata.get("per_key_transform", {}).get(f"{layer}|{mode}")
        if rec and isinstance(rec, dict) and "variants" in rec:
            # choose selected or default
            vid = reducer_select or rec.get("default")
            if not vid:
                raise FileNotFoundError(f"No default reducer variant recorded for ({layer}, {mode}).")
            # selected might be stored under canonical path if historically default; otherwise look for variant file
            vpath = self._embeddings_file_path(layer, mode, variant=vid)
            if not os.path.isfile(vpath):
                raise FileNotFoundError(f"Cached tensor not found for reducer '{vid}': {vpath}")
            return load_file(vpath)["embedding"]

        # fallback to original base file
        path = self._embeddings_file_path(layer, mode, variant=None)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cached tensor not found: {path}")
        return load_file(path)["embedding"]

    def delete_cache(self) -> List[str]:
        """
        Delete all cached tensors and metadata. Returns list of removed paths.
        """
        removed = []
        for root in [self.input_dir, self.output_dir, self.transforms_dir]:
            if os.path.isdir(root):
                for f in os.listdir(root):
                    fp = os.path.join(root, f)
                    if os.path.isfile(fp):
                        try:
                            os.remove(fp)
                            removed.append(fp)
                        except Exception:
                            pass
        for fp in [self.y_path, self.final_output_path, self.meta_path]:
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                    removed.append(fp)
                except Exception:
                    pass
        return removed

    def rehash(self) -> bool:
        """
        Recompute model fingerprint and compare to stored one.
        Returns True if unchanged, False if model parameters/structure differ.
        """
        return self._model_fingerprint(self.model) == self.model_fp

    # ------------ Static / Internal Helpers ------------

    @staticmethod
    def _hash_bytes(buf: bytes) -> str:
        h = hashlib.sha256()
        h.update(buf)
        return h.hexdigest()

    @staticmethod
    @torch.no_grad()
    def _model_fingerprint(model: nn.Module) -> str:
        model.eval()
        h = hashlib.sha256()
        h.update(model.__class__.__name__.encode())
        for k, v in model.state_dict().items():
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    @staticmethod
    @torch.no_grad()
    def _dataset_fingerprint(dataloader, batches: int) -> str:
        m = hashlib.sha256()
        total = len(dataloader)
        take = total if batches <= 0 else min(total, batches)

        for i, batch in enumerate(dataloader):
            if i >= take:
                break
            sample = batch
            if torch.is_tensor(sample):
                m.update(sample.detach().cpu().numpy().tobytes())
            elif isinstance(sample, (list, tuple)):
                for item in sample:
                    if torch.is_tensor(item):
                        m.update(item.detach().cpu().numpy().tobytes())
                    else:
                        m.update(str(item).encode())
            else:
                raise TypeError("Unsupported batch type for fingerprinting")
        return m.hexdigest()

    @staticmethod
    def _normalize_layers_modes(layer_names, capture_modes):
        if isinstance(layer_names, str):
            layer_list = [layer_names]
        else:
            layer_list = list(layer_names)
        if isinstance(capture_modes, str):
            mode_list = [capture_modes] * len(layer_list)
        else:
            mode_list = list(capture_modes)
            if len(mode_list) != len(layer_list):
                raise ValueError("capture_modes length must match target_layer_names length")
        for m in mode_list:
            if m not in ("input", "output", "both"):
                raise ValueError("capture_modes must be 'input','output', or 'both'")
        return layer_list, mode_list

    @staticmethod
    def _ensure_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    # ------------ Reducer persistence (safetensors) ------------

    def _safe_name(self, s: str) -> str:
        # keep dots and dashes but replace other unsafe chars
        return re.sub(r'[^a-zA-Z0-9_.-]+', '_', s)

    def _embeddings_file_path(self, layer: str, mode: str, variant: Optional[str]) -> str:
        root = self.input_dir if mode == "input" else self.output_dir
        fname = f"{layer}.safetensors" if not variant else f"{layer}__{self._safe_name(variant)}.safetensors"
        return os.path.join(root, fname)

    def _transform_file_path(self, layer: str, mode: str, variant: Optional[str] = None) -> str:
        base = f"{self._safe_name(layer)}__{mode}"
        if variant:
            base += f"__{self._safe_name(variant)}"
        return os.path.join(self.transforms_dir, f"{base}.safetensors")

    def _reducers_public(self, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pub = []
        for s in specs or []:
            d = {k: s[k] for k in ("id", "name", "kwargs", "fit_batches", "min", "max") if k in s}
            pub.append(d)
        return pub

    # normalize multi-reducer inputs into list of specs (+ optional class builder)
    def _normalize_reducers(self) -> List[Dict[str, Any]]:
        names = self.reducer_name
        if not names:
            return []
        names_list = names if isinstance(names, list) else [names]
        rk = self.reducer_kwargs
        fitb = self.reducer_fit_batches
        rdt = self.reducer_dim_threshold

        # helper: pick override builder by index
        def builder_for(idx: int) -> Optional[Any]:
            co = self.class_overrides
            if co is None:
                return None
            if isinstance(co, (list, tuple)):
                return co[idx] if idx < len(co) else None
            return co

        def kwargs_for(name: str, idx: int) -> Dict[str, Any]:
            if isinstance(rk, list):
                return rk[idx] if idx < len(rk) else {}
            if isinstance(rk, dict) and name in rk:
                return rk[name] or {}
            if isinstance(rk, dict):
                return rk
            return {}

        def fit_for(idx: int) -> int:
            if isinstance(fitb, list):
                return int(fitb[idx]) if idx < len(fitb) else 0
            return int(fitb or 0)

        def range_for(idx: int) -> Tuple[int, Optional[int]]:
            val = rdt[idx] if isinstance(rdt, list) and idx < len(rdt) else (rdt if not isinstance(rdt, list) else None)
            if isinstance(val, tuple) and len(val) == 2:
                rmin, rmax = int(val[0]), (None if val[1] is None else int(val[1]))
                return rmin, rmax
            if isinstance(val, int):
                return int(val), None
            return 10 ** 12, None

        # guard against duplicate names without overrides
        name_to_idx: Dict[str, List[int]] = {}
        for i, nm in enumerate(names_list):
            name_to_idx.setdefault(str(nm), []).append(i)
        dup_names = [nm for nm, idxs in name_to_idx.items() if len(idxs) > 1]
        if dup_names:
            for nm in dup_names:
                idxs = name_to_idx[nm]
                # Require at least one override among duplicates of this name
                if not any(builder_for(i) is not None for i in idxs):
                    raise ValueError(
                        f"Duplicate reducer_name '{nm}' provided multiple times without class_overrides. "
                        f"Provide class_overrides for those entries or use unique reducer names."
                    )

        specs: List[Dict[str, Any]] = []
        for i, nm in enumerate(names_list):
            rmin, rmax = range_for(i)
            builder = builder_for(i)
            spec = {
                "id": str(nm),
                "name": str(nm),
                "kwargs": kwargs_for(str(nm), i),
                "fit_batches": fit_for(i),
                "min": int(rmin),
                "max": (None if rmax is None else int(rmax))
            }
            if builder is not None:
                spec["builder"] = builder
            specs.append(spec)

        # Ensure unique IDs when same reducer name is repeated
        counts: Dict[str, int] = {}
        for s in specs:
            base = s["id"]
            c = counts.get(base, 0)
            if c > 0:
                s["id"] = f"{base}#{c + 1}"
            counts[base] = c + 1
        return specs

    def _reconcile_metadata_filesystem(self):
        """
        Minimal reconciliation: ensure required keys exist to avoid AttributeError.
        """
        self.metadata.setdefault("layer_modes", {})
        self.metadata.setdefault("shapes", {})
        self.metadata.setdefault("dtypes", {})
        self.metadata.setdefault("transform_defaults", {})
        self.metadata.setdefault("per_key_transform", {})

    def _build_reducer(self, spec: Dict[str, Any]) -> Any:
        builder = spec.get("builder", None)
        kwargs = spec.get("kwargs") or {}
        if builder is not None:
            if isinstance(builder, nn.Module):  # why should the speec be a nn module????
                return copy.deepcopy(builder)
            try:
                return builder(**kwargs)
            except TypeError:
                return builder()
        return create_feature_reducer(spec["name"], **kwargs)

    def _save_reducer(self, layer: str, mode: str, reducer: nn.Module, input_dim: int,
                      variant: Optional[str] = None, reg_name: Optional[str] = None,
                      kwargs: Optional[Dict[str, Any]] = None, default_variant: bool = False):
        """
        Save a fitted reducer for (layer, mode[, variant]) to safetensors and record metadata.
        """
        if reducer is None:
            return
        os.makedirs(self.transforms_dir, exist_ok=True)
        vid = str(variant or (reg_name or reducer.__class__.__name__.lower()))
        path = self._transform_file_path(layer, mode, variant=vid)

        state = reducer.state_dict()
        tensors = {k: v.detach().cpu() for k, v in state.items()}
        try:
            save_file(tensors, path)
        except ValueError as e:
            if "non contiguous tensor" in str(e):
                tensors = {k: v.contiguous() for k, v in tensors.items()}
                save_file(tensors, path)
            else:
                raise

        key = f"{layer}|{mode}"
        pkt = self.metadata.setdefault("per_key_transform", {})
        rec = pkt.setdefault(key, {})
        if "variants" not in rec:
            rec.clear()
            rec["variants"] = {}
        vrec = {
            "class": reducer.__class__.__name__,
            "name": reg_name or vid,
            "kwargs": kwargs or {},
            "path": path,
            "input_dim": int(input_dim),
        }
        rec["variants"][vid] = vrec
        if default_variant or "default" not in rec:
            rec["default"] = vid
        self.metadata["updated_unix"] = time.time()
        self._save_metadata()

    # load saved reducer
    def _load_reducer_for_key(self, layer: str, mode: str, variant: Optional[str] = None) -> Optional[nn.Module]:
        # look up in registry and get values to create reducer
        rec = self.metadata.get("per_key_transform", {}).get(f"{layer}|{mode}")
        if not rec:
            return None
        if isinstance(rec, dict) and "variants" in rec:
            vid = variant or rec.get("default")
            if not vid:
                return None
            vrec = rec["variants"].get(vid)
            if not vrec:
                return None
            path = vrec.get("path") or self._transform_file_path(layer, mode, variant=vid)
            if not os.path.isfile(path):
                return None
            name = vrec.get("name")
            kwargs = vrec.get("kwargs") or {}
            # use registry to instantiate; assumes compatible with saved state
            reducer = create_feature_reducer(name, **kwargs)
            state_tensors = load_file(path)
            reducer.load_state_dict(state_tensors, strict=False)
            return reducer
        return None

    # compute applicable reducers based on the dim.
    def _applicable_reducers(self, sample_dim: int, reducer_select: Optional[str] = None) -> List[Dict[str, Any]]:
        specs = self._reducers_norm or []
        applicable = [s for s in specs if
                      sample_dim > int(s["min"]) and (s["max"] is None or sample_dim <= int(s["max"]))]
        if reducer_select:
            applicable = [s for s in applicable if reducer_select in (s["id"], s["name"])]
            if not applicable:
                raise RuntimeError(
                    f"Selected reducer '{reducer_select}' is not applicable for sample_dim={sample_dim}.")
        if not applicable:
            finite_max = [int(s["max"]) for s in specs if s["max"] is not None]
            if finite_max and all(sample_dim > m for m in finite_max):
                raise RuntimeError(
                    f"No reducer applies for sample_dim={sample_dim}: exceeds all configured max ranges.")
        return applicable

    # ------------ Internal compute/store ------------

    @torch.no_grad()
    def _store_tensor(self, path: str, tensor: torch.Tensor):
        save_file({"embedding": tensor.cpu()}, path)

    @torch.no_grad()
    def _update_layer_metadata(self, layer: str):
        """
        After writing layer files, update metadata layer_modes to most complete state.
        Variant-aware: if a default reducer variant exists, shapes/dtypes are taken from it.
        """

        # Determine existing availability using variant-aware paths
        def has_mode(m: str) -> Tuple[bool, Optional[str]]:
            base = self._embeddings_file_path(layer, m, None)
            if os.path.isfile(base):
                return True, base
            rec = self.metadata.get("per_key_transform", {}).get(f"{layer}|{m}")
            if rec and "default" in rec:
                v = rec["default"]
                vpath = self._embeddings_file_path(layer, m, v)
                if os.path.isfile(vpath):
                    return True, vpath
            return False, None

        have_input, in_path = has_mode("input")
        have_output, out_path = has_mode("output")
        if have_input and have_output:
            mode = "both"
        elif have_input:
            mode = "input"
        elif have_output:
            mode = "output"
        else:
            return
        self.metadata["layer_modes"][layer] = mode

        if have_input and in_path:
            tin = load_file(in_path)["embedding"]
            self.metadata.setdefault("shapes", {}).setdefault(layer, {})["input"] = list(tin.shape)
            self.metadata.setdefault("dtypes", {}).setdefault(layer, {})["input"] = str(tin.dtype)
        if have_output and out_path:
            tout = load_file(out_path)["embedding"]
            self.metadata.setdefault("shapes", {}).setdefault(layer, {})["output"] = list(tout.shape)
            self.metadata.setdefault("dtypes", {}).setdefault(layer, {})["output"] = str(tout.dtype)

    # helper to find missing embeddings
    def _compute_missing(self, layers: List[str], modes: List[str], reducer_select: Optional[str] = None) -> Dict[
        str, List[str]]:
        """
        Determine which (layer, mode) pairs are missing by checking actual files on disk (not just metadata),
        with awareness of reducer variants. If variants exist, check selected or default variant files.
        """
        needed: Dict[str, set] = {}
        for layer, mode in zip(layers, modes):
            requested = {"input", "output"} if mode == "both" else {mode}
            have_set = set()
            for m in requested:
                rec = self.metadata.get("per_key_transform", {}).get(f"{layer}|{m}")
                if rec and "variants" in rec:
                    vid = reducer_select or rec.get("default")
                    if vid and os.path.isfile(self._embeddings_file_path(layer, m, variant=vid)):
                        have_set.add(m)
                else:
                    if os.path.isfile(self._embeddings_file_path(layer, m, variant=None)):
                        have_set.add(m)
            missing_for = requested - have_set
            if missing_for:
                needed.setdefault(layer, set()).update(missing_for)
        # keep order deterministic and mode order stable
        return {layer: [m for m in ("input", "output") if m in modes_set] for layer, modes_set in needed.items()}

    @torch.no_grad()
    def _compute_and_store(self, missing: Dict[str, List[str]], all_requested_layers: List[str],
                           reducer_select: Optional[str] = None):
        """
        Run the model once over the dataset (respecting max_batches) to compute missing embeddings,
        final outputs and y (if needed). Apply reducer variants when applicable.
        Can apply multipel reducers so data does not have to be read multiple times.
        """
        if not missing and os.path.isfile(self.y_path) and os.path.isfile(self.final_output_path):
            return  # nothing to do

        # Build wrapper target lists (only for missing (layer,mode))
        capture_layers = []
        capture_modes = []
        entry_indices = []
        for layer, mode_list in missing.items():
            for m in mode_list:
                capture_layers.append(layer)
                capture_modes.append(m)
                entry_indices.append(0)  # default

        wrapper = None
        if capture_layers:
            wrapper = ModelInputOutputWrapper(
                self.model,
                target_layer_names=capture_layers,
                flatten=False,
                concat=False,
                capture_modes=capture_modes,
                entry_indices=entry_indices,
            )

        device = self.device or next(self.model.parameters()).device
        self.model.eval()

        # Accumulators
        acc_default: Dict[Tuple[str, str], List[torch.Tensor]] = {(layer, mode): [] for layer, modes in missing.items()
                                                                  for mode in modes}
        acc_variants: Dict[Tuple[str, str, str], List[torch.Tensor]] = {}
        acc_final: List[torch.Tensor] = []
        acc_y: List[torch.Tensor] = []

        # Per-(layer, mode) state: shared buffer + per-variant state
        transform_states: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Determine tqdm total (respect self.max_batches)
        try:
            total_batches = len(self.dataloader)
        except Exception:
            total_batches = None
        if self.max_batches is not None and total_batches is not None:
            total = min(self.max_batches, total_batches)
        else:
            total = self.max_batches if self.max_batches is not None else total_batches

        with torch.no_grad():
            dl_iter = tqdm(self.dataloader,
                           total=None if self.max_batches is None else min(self.max_batches, len(self.dataloader)),
                           desc="Computing embeddings", unit="batch")
            for b_idx, batch in enumerate(dl_iter):
                if self.max_batches is not None and b_idx >= self.max_batches:
                    break
                # Extract x, y
                if isinstance(batch, dict):
                    x = batch.get("x")
                    y = batch.get("y")
                elif isinstance(batch, (list, tuple)):
                    x = batch[0]
                    y = batch[1] if len(batch) > 1 else None
                else:
                    x, y = batch, None

                x = x.to(device)
                if y is not None and torch.is_tensor(y):
                    y = y.to(device)

                if wrapper:
                    # wrapper returns intermediate and final outputs
                    inter_list, final_out = wrapper(x)
                    for i, (layer, mode) in enumerate(zip(capture_layers, capture_modes)):
                        key = (layer, mode)
                        if key not in acc_default:
                            continue
                        t = inter_list[i]

                        st = transform_states.get(key)
                        # get applicable reducer
                        if st is None:
                            try:
                                sample_dim = int(torch.tensor(t.shape[1:]).prod().item()) if t.dim() > 1 else 1
                            except AttributeError as e:
                                print(f"DEBUG: Caught AttributeError in embedding_cache._compute_and_store.")
                                print(f"  - Layer: {layer}, Mode: {mode}, Reducer Select: {reducer_select}")
                                print(f"  - Problematic value 't' is of type {type(t)}")
                                if isinstance(t, (list, tuple)):
                                    for i, item in enumerate(t):
                                        print(
                                            f"    - Item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}")
                                else:
                                    print(f"  - Value: {t}")
                                raise e
                            applicable = self._applicable_reducers(sample_dim, reducer_select=reducer_select)
                            st = {
                                "sample_dim": sample_dim,
                                "has_any": bool(applicable),
                                "buffer": [],  # shared buffer of original tensors (CPU)
                                "seen": 0,
                                "applicable": []  # per-variant states
                            }
                            if applicable:
                                for j, spec in enumerate(applicable):
                                    reducer = self._build_reducer(spec)  # CHANGED
                                    if isinstance(reducer, nn.Module):
                                        reducer = reducer.to(device)
                                    st["applicable"].append({
                                        "vid": str(spec["id"]),
                                        "spec": spec,
                                        "reducer": reducer,
                                        "fit_batches": int(spec.get("fit_batches", 0) or 0),
                                        "initialized": False,
                                        "fitted": False,
                                        "last_idx": 0
                                    })
                            transform_states[key] = st

                        if not st["has_any"]:
                            # No reducer applies (min-side) -> store original base
                            if reducer_select:
                                # selection requested but none applicable
                                raise RuntimeError(
                                    f"No reducer available for selection '{reducer_select}' on ({layer}, {mode}).")
                            acc_default[key].append(t.detach().cpu())
                            continue

                        # Shared buffering for all reducers
                        st["buffer"].append(t.detach().cpu())
                        st["seen"] += 1

                        # Initialize/fit reducers as they become ready, and transform new portion of buffer
                        for j, vst in enumerate(st["applicable"]):
                            reducer = vst["reducer"]
                            just_initialized = False
                            # Some reducer require fitting. These specify how much they require. Once the requested data amount is reached it is fitted.
                            if not vst["initialized"]:
                                fb = vst["fit_batches"]
                                if fb > 0 and st["seen"] >= fb and not vst["fitted"]:
                                    # Fit using the entire buffered data
                                    buf_cat = torch.cat(st["buffer"], dim=0).to(device).float()
                                    if hasattr(reducer, "fit") and callable(getattr(reducer, "fit")):
                                        reducer.fit(buf_cat)
                                    vst["fitted"] = True
                                    vst["initialized"] = True
                                    just_initialized = True
                                    # Save reducer state and mark default for first applicable
                                    self._save_reducer(layer, mode, reducer, st["sample_dim"],
                                                       variant=vst["vid"], reg_name=vst["spec"]["name"],
                                                       kwargs=vst["spec"].get("kwargs") or {},
                                                       default_variant=(j == 0))  # simplified
                                elif fb == 0:
                                    # Immediate initialize (optionally one-shot fit on current batch)
                                    if hasattr(reducer, "fit") and callable(getattr(reducer, "fit")):
                                        reducer.fit(t.to(device).float())
                                    vst["fitted"] = True
                                    vst["initialized"] = True
                                    just_initialized = True
                                    self._save_reducer(layer, mode, reducer, st["sample_dim"],
                                                       variant=vst["vid"], reg_name=vst["spec"]["name"],
                                                       kwargs=vst["spec"].get("kwargs") or {},
                                                       default_variant=(j == 0))  # simplified

                            if vst["initialized"]:
                                # Transform only the unprocessed tail of the shared buffer
                                tail_start = vst["last_idx"]
                                if just_initialized and tail_start == 0:
                                    # First time: transform full buffer at once
                                    buf_cat = torch.cat(st["buffer"], dim=0).to(device).float()
                                    outb = reducer(buf_cat).detach().cpu()
                                    acc_variants.setdefault((layer, mode, vst["vid"]), []).append(outb)
                                    vst["last_idx"] = len(st["buffer"])
                                else:
                                    # Transform only the newest item (current batch)
                                    if len(st["buffer"]) > tail_start:
                                        new_items = st["buffer"][tail_start:]
                                        if len(new_items) == 1:
                                            out = reducer(new_items[0].to(device).float()).detach().cpu()
                                        else:
                                            out = reducer(torch.cat(new_items, dim=0).to(device).float()).detach().cpu()
                                        acc_variants.setdefault((layer, mode, vst["vid"]), []).append(out)
                                        vst["last_idx"] = len(st["buffer"])
                else:
                    final_out = self.model(x)

                acc_final.append(final_out.detach().cpu())
                if y is not None:
                    acc_y.append(y.detach().cpu())

        # Final flush: for any reducer that still has untransformed tail
        for (layer, mode), st in transform_states.items():
            if not st.get("has_any"):
                continue
            for j, vst in enumerate(st["applicable"]):
                if not vst["initialized"]:
                    continue
                tail_start = vst["last_idx"]
                if tail_start < len(st["buffer"]):
                    reducer = vst["reducer"]
                    tail = st["buffer"][tail_start:]
                    if len(tail) == 1:
                        out = reducer(tail[0].to(device).float()).detach().cpu()
                    else:
                        out = reducer(torch.cat(tail, dim=0).to(device).float()).detach().cpu()
                    acc_variants.setdefault((layer, mode, vst["vid"]), []).append(out)
                    vst["last_idx"] = len(st["buffer"])

        # Write embeddings: base for originals only; variants separately (with name in filename)
        for (layer, mode), tensor_list in acc_default.items():
            if not tensor_list:
                continue
            full = torch.cat(tensor_list, dim=0)
            out_path = self._embeddings_file_path(layer, mode, variant=None)
            self._store_tensor(out_path, full)
            self._update_layer_metadata(layer)

        for (layer, mode, vid), tensor_list in acc_variants.items():
            if not tensor_list:
                continue
            full = torch.cat(tensor_list, dim=0)
            out_path = self._embeddings_file_path(layer, mode, variant=vid)
            self._store_tensor(out_path, full)
            self._update_layer_metadata(layer)

        # Write final outputs / y if not present (unchanged)
        if acc_final and not os.path.isfile(self.final_output_path):
            final_full = torch.cat(acc_final, dim=0)
            self._store_tensor(self.final_output_path, final_full)
            self.metadata.setdefault("shapes", {}).setdefault("final_output", {})["output"] = list(final_full.shape)
            self.metadata.setdefault("dtypes", {}).setdefault("final_output", {})["output"] = str(final_full.dtype)

        # Write y if provided and not present
        if acc_y and not os.path.isfile(self.y_path):
            y_full = torch.cat(acc_y, dim=0)
            self._store_tensor(self.y_path, y_full)
            self.metadata.setdefault("shapes", {}).setdefault("y", {})["y"] = list(y_full.shape)
            self.metadata.setdefault("dtypes", {}).setdefault("y", {})["y"] = str(y_full.dtype)

        self.metadata["updated_unix"] = time.time()
        self._save_metadata()

    # This is a helper function to to create a wrapper reading out intermediate values. This can be used to compute multiple embedding
    # but more importantly we can use it to create models to compute the confidence efficently if we need it for multiple layers like mahalanobis.
    def make_wrapper(self,
                     target_layer_names,
                     capture_modes='output',
                     flatten=False,
                     concat=False,
                     entry_indices=0,
                     return_final: bool = True,
                     return_y: bool = False,
                     reducer_select: Optional[str] = None) -> ModelInputOutputWrapper:
        """
        Construct a ModelInputOutputWrapper. If saved reducers exist, apply selected/default variant in-flight.
        """
        layers, modes = self._normalize_layers_modes(target_layer_names, capture_modes)
        single_entry = isinstance(target_layer_names, str)
        layers_arg = layers if not single_entry else layers[0]

        # Load per-key reducers (variant-aware)
        feature_reducers: Dict[Tuple[str, str], nn.Module] = {}
        expanded = []
        for l, m in zip(self._ensure_list(layers), modes):
            if m == "both":
                expanded.append((l, "input"))
                expanded.append((l, "output"))
            else:
                expanded.append((l, m))
        device = self.device or next(self.model.parameters()).device
        for key in expanded:
            reducer = self._load_reducer_for_key(*key, variant=reducer_select)
            if reducer is not None:
                feature_reducers[key] = reducer.to(device)

        return ModelInputOutputWrapper(
            self.model,
            target_layer_names=layers_arg,
            capture_modes=modes,
            flatten=flatten,
            concat=concat,
            entry_indices=entry_indices if isinstance(entry_indices, (list, tuple)) else entry_indices,
            return_final=return_final,
            return_y=return_y,
            feature_reducers=feature_reducers if feature_reducers else None
        )

    # direct compute emthod with store.
    def _compute_direct(self,
                        target_layer_names,
                        capture_modes='output',
                        flatten=False,
                        concat=False,
                        entry_indices=0,
                        return_y=True,
                        return_final=True,
                        reducer_select: Optional[str] = None):
        """
        Compute embeddings/final/y without writing files or metadata.
        Applies the selected reducer variant if available; otherwise the first applicable one.
        If no reducer applies due to max overflow, raises; if none applies due to min-side, returns original.
        """
        device = self.device or next(self.model.parameters()).device
        layers, modes = self._normalize_layers_modes(target_layer_names, capture_modes)
        single_entry = isinstance(target_layer_names, str)
        layers_arg = layers if not single_entry else layers[0]

        wrapper = ModelInputOutputWrapper(
            self.model,
            target_layer_names=layers_arg,
            capture_modes=modes,
            flatten=False,
            concat=False,
            entry_indices=entry_indices if isinstance(entry_indices, (list, tuple)) else entry_indices,
            return_final=return_final,
            return_y=return_y
        )

        # Accumulators
        inter_acc: Optional[List[List[torch.Tensor]]] = None  # per expanded output
        final_acc = [] if return_final else None
        y_acc = [] if return_y else None

        # Per-expanded-key reducer state (only selected/default variant is computed)
        reducer_states: List[Optional[Dict[str, Any]]] = None  # one entry per expanded output

        self.model.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(self.dataloader):
                if self.max_batches is not None and b_idx >= self.max_batches:
                    break
                if isinstance(batch, dict):
                    x = batch.get("x")
                    y = batch.get("y")
                elif isinstance(batch, (list, tuple)):
                    x = batch[0]
                    y = batch[1] if len(batch) > 1 else None
                else:
                    x, y = batch, None
                x = x.to(device)
                if y is not None and torch.is_tensor(y):
                    y = y.to(device)

                # Wrapper call
                if return_final and return_y:
                    inter_list, final_out, y_batch = wrapper(x, y)
                elif return_final:
                    inter_list, final_out = wrapper(x, y)
                    y_batch = None
                elif return_y:
                    inter_list, y_batch = wrapper(x, y)
                    final_out = None
                else:
                    inter_list = wrapper(x, y)
                    final_out = None
                    y_batch = None

                # Initialize accumulators and reducer_states on first batch
                if isinstance(inter_list, torch.Tensor):
                    inter_list = [inter_list]
                if inter_acc is None:
                    inter_acc = [[] for _ in range(len(inter_list))]
                    reducer_states = [None for _ in range(len(inter_list))]

                # Per expanded output
                for i, t in enumerate(inter_list):
                    st = reducer_states[i]
                    if st is None:
                        sample_dim = int(torch.tensor(t.shape[1:]).prod().item()) if t.dim() > 1 else 1
                        applicable = self._applicable_reducers(sample_dim, reducer_select=reducer_select)
                        if not applicable:
                            # No reducer applicable due to min-side: keep originals unless selection demanded
                            if reducer_select:
                                raise RuntimeError(
                                    f"No reducer available for selection '{reducer_select}' for expanded index {i}.")
                            reducer_states[i] = {"apply": False}
                            st = reducer_states[i]
                        else:
                            # Choose selected or default (first applicable)
                            chosen = None
                            if reducer_select:
                                for spec in applicable:
                                    if reducer_select in (spec["id"], spec["name"]):
                                        chosen = spec
                                        break
                            if chosen is None:
                                chosen = applicable[0]
                            reducer = self._build_reducer(chosen)  # CHANGED
                            if isinstance(reducer, nn.Module):
                                reducer = reducer.to(device)
                            reducer_states[i] = {
                                "apply": True,
                                "spec": chosen,
                                "reducer": reducer,
                                "fit_batches": int(chosen.get("fit_batches", 0) or 0),
                                "buffer": [],
                                "seen": 0,
                                "initialized": False,
                                "last_idx": 0
                            }
                            st = reducer_states[i]

                    if not st.get("apply", False):
                        inter_acc[i].append(t.detach().cpu())
                        continue

                    # Shared buffer for this expanded output
                    st["buffer"].append(t.detach().cpu())
                    st["seen"] += 1
                    reducer = st["reducer"]
                    fb = st["fit_batches"]
                    just_initialized = False

                    if not st["initialized"]:
                        if fb > 0 and st["seen"] >= fb:
                            buf_cat = torch.cat(st["buffer"], dim=0).to(device).float()
                            if hasattr(reducer, "fit") and callable(getattr(reducer, "fit")):
                                reducer.fit(buf_cat)
                            st["initialized"] = True
                            just_initialized = True
                        elif fb == 0:
                            if hasattr(reducer, "fit") and callable(getattr(reducer, "fit")):
                                reducer.fit(t.to(device).float())
                            st["initialized"] = True
                            just_initialized = True

                    if st["initialized"]:
                        tail_start = st["last_idx"]
                        if just_initialized and tail_start == 0:
                            buf_cat = torch.cat(st["buffer"], dim=0).to(device).float()
                            outb = reducer(buf_cat).detach().cpu()
                            inter_acc[i].append(outb)
                            st["last_idx"] = len(st["buffer"])
                        else:
                            if len(st["buffer"]) > tail_start:
                                new_items = st["buffer"][tail_start:]
                                if len(new_items) == 1:
                                    out = reducer(new_items[0].to(device).float()).detach().cpu()
                                else:
                                    out = reducer(torch.cat(new_items, dim=0).to(device).float()).detach().cpu()
                                inter_acc[i].append(out)
                                st["last_idx"] = len(st["buffer"])

                if return_final and final_out is not None:
                    final_acc.append(final_out.detach().cpu())
                if return_y and y_batch is not None:
                    y_acc.append(y_batch.detach().cpu())

        # Concatenate outputs
        inter_cat = [torch.cat(v, dim=0) for v in inter_acc] if inter_acc else []
        if flatten:
            inter_cat = [t.view(t.size(0), -1) for t in inter_cat]
        if concat:
            embeddings = torch.cat(inter_cat, dim=-1) if inter_cat else None
        else:
            embeddings = inter_cat[0] if len(inter_cat) == 1 else inter_cat

        final_tensor = torch.cat(final_acc, dim=0) if (return_final and final_acc) else None
        y_tensor = torch.cat(y_acc, dim=0) if (return_y and y_acc) else None

        if return_final and return_y:
            return embeddings, final_tensor, y_tensor
        if return_final:
            return embeddings, final_tensor
        if return_y:
            return embeddings, y_tensor
        del reducer_states
        return embeddings

    # gets all embeddings
    @torch.no_grad()
    def get_embeddings(self,
                       target_layer_names,
                       capture_modes='output',
                       flatten=False,
                       concat=False,
                       entry_indices=0,
                       return_y=True,
                       return_final=True,
                       reducer_select: Optional[str] = None,
                       store: bool = True, verify=False):
        """
        Ensure embeddings (and optionally final outputs, y) are cached, then return.
        If reducers apply for a key, original data is not stored; embeddings are stored as layer__{reducer}.safetensors.
        When reducer_select is None, the default (first applicable) is used; otherwise the named reducer is used.
        """
        if not store:
            return self._compute_direct(
                target_layer_names=target_layer_names,
                capture_modes=capture_modes,
                flatten=flatten,
                concat=concat,
                entry_indices=entry_indices,
                return_y=return_y,
                return_final=return_final,
                reducer_select=reducer_select
            )

        single_entry = isinstance(target_layer_names, str)

        layers, modes = self._normalize_layers_modes(target_layer_names, capture_modes)
        missing = self._compute_missing(layers, modes, reducer_select=reducer_select)
        if missing:
            self._compute_and_store(missing, layers, reducer_select=reducer_select)
        else:
            if return_final and not os.path.isfile(self.final_output_path):
                self._compute_and_store({}, layers, reducer_select=reducer_select)
            if return_y and not os.path.isfile(self.y_path):
                self._compute_and_store({}, layers, reducer_select=reducer_select)
        if verify:
            self._verify_cached_embeddings_single_batch(layers, modes, reducer_select=reducer_select)

        tensors = []
        for layer, mode in zip(layers, modes):
            mode_list = ["input", "output"] if mode == "both" else [mode]
            for m in mode_list:
                t = self._load_embeddings_from_disk(layer, m, reducer_select=reducer_select)
                if flatten:
                    t = t.view(t.size(0), -1)
                tensors.append(t)

        if concat:
            emb = torch.cat(tensors, dim=-1)
        else:
            emb = tensors[0] if single_entry and len(tensors) == 1 else tensors

        final_out = None
        if return_final:
            final_out = load_file(self.final_output_path)["embedding"]

        y_tensor = None
        if return_y and os.path.isfile(self.y_path):
            y_tensor = load_file(self.y_path)["embedding"]

        if return_final and return_y:
            return emb, final_out, y_tensor
        if return_final:
            return emb, final_out
        if return_y:
            return emb, y_tensor
        return emb

    # only gets embeddings that are correctly classified. This is done by the cache also string the final outputs.(requires classification model)
    @torch.no_grad()
    def get_correct_embeddings(cache, target_layer_names,
                               capture_modes='output',
                               flatten=False, concat=False,
                               entry_indices=0, reducer_select: Optional[str] = None):
        """
        Return embeddings (filtered to correctly-classified examples),
        plus filtered final outputs and labels.

        - cache: LayerEmbeddingCache instance
        - target_layer_names, capture_modes: passed to cache.get_embeddings
        """
        # Request embeddings plus final outputs and y
        emb, final_out, y = cache.get_embeddings(
            target_layer_names=target_layer_names,
            capture_modes=capture_modes,
            flatten=flatten,
            concat=concat,
            entry_indices=entry_indices,
            return_final=True,
            return_y=True, reducer_select=reducer_select
        )

        # Determine predictions (assumes final_out are logits or class-probs)
        # If final_out is 2D: (N, num_classes)
        preds = final_out.argmax(dim=-1)

        # Boolean mask of correct predictions
        correct_mask = preds == y

        # Helper to index embedding tensors (support list or single tensor)
        def _index_embeddings(e: Union[List[torch.Tensor], torch.Tensor]):
            # Ensure mask shape matches embedding shape
            if isinstance(e, torch.Tensor):
                if e.shape[0] != correct_mask.shape[0]:
                    print("Shape error in _index_embeddings:")
                    raise ValueError(f"Embedding shape {e.shape} does not match mask shape {correct_mask.shape}. "
                                     "This usually means the wrong reducer variant or flatten/concat settings were used.")
                return e[correct_mask]
            else:
                return [t[correct_mask] if t.shape[0] == correct_mask.shape[0] else
                        (_raise_shape_error(t, correct_mask)) for t in e]

        def _raise_shape_error(t, mask):
            print("Shape error in _index_embeddings:")
            raise ValueError(f"Embedding shape {t.shape} does not match mask shape {mask.shape}. "
                             "This usually means the wrong reducer variant or flatten/concat settings were used.")

        emb_correct = _index_embeddings(emb)
        final_correct = final_out[correct_mask]
        y_correct = y[correct_mask]

        return emb_correct, final_correct, y_correct

    def __call__(self,
                 target_layer_names,
                 capture_modes='input',
                 flatten=False,
                 concat=False,
                 return_y=True,
                 return_final=True,
                 entry_indices=0,
                 reducer_select: Optional[str] = None,
                 store: bool = True):
        return self.get_embeddings(
            target_layer_names=target_layer_names,
            capture_modes=capture_modes,
            flatten=flatten,
            concat=concat,
            entry_indices=entry_indices,
            return_y=return_y,
            return_final=return_final,
            reducer_select=reducer_select,
            store=store
        )

    # TODO what is the differnce to compute direct?
    def run_wrapper(self,
                    target_layer_names,
                    capture_modes='output',
                    flatten=False,
                    concat=False,
                    entry_indices=0,
                    return_y=True,
                    return_final=True,
                    reducer_select: Optional[str] = None):
        """
        Direct  computation over the full dataloader using ModelInputOutputWrapper.
        """
        wrapper = self.make_wrapper(
            target_layer_names=target_layer_names,
            capture_modes=capture_modes,
            flatten=flatten,
            concat=concat,
            entry_indices=entry_indices,
            return_final=return_final,
            return_y=return_y,
            reducer_select=reducer_select
        )
        device = self.device or next(self.model.parameters()).device
        self.model.eval()

        # Accumulators
        inter_acc = None  # list of lists (per expanded output) or single list
        final_acc = [] if return_final else None
        y_acc = [] if return_y else None

        with torch.no_grad():
            for b_idx, batch in enumerate(self.dataloader):
                if self.max_batches is not None and b_idx >= self.max_batches:
                    break
                if isinstance(batch, dict):
                    x = batch.get("x")
                    y = batch.get("y")
                elif isinstance(batch, (list, tuple)):
                    x = batch[0]
                    y = batch[1] if len(batch) > 1 else None
                else:
                    x, y = batch, None
                x = x.to(device)
                if y is not None and torch.is_tensor(y):
                    y = y.to(device)

                # Wrapper returns variable structure based on flags; extract embeddings first.
                if return_final and return_y:
                    inter, final_out, y_batch = wrapper(x, y)
                elif return_final:
                    inter, final_out = wrapper(x, y)
                    y_batch = None
                elif return_y:
                    inter, y_batch = wrapper(x, y)
                    final_out = None
                else:
                    inter = wrapper(x, y)
                    final_out = None
                    y_batch = None

                # Normalize embeddings to list
                if isinstance(inter, torch.Tensor):
                    inter_list = [inter]
                else:
                    inter_list = inter
                if inter_acc is None:
                    inter_acc = [[] for _ in range(len(inter_list))]
                for i, t in enumerate(inter_list):
                    inter_acc[i].append(t.detach().cpu())

                if return_final and final_out is not None:
                    final_acc.append(final_out.detach().cpu())
                if return_y and y_batch is not None:
                    y_acc.append(y_batch.detach().cpu())

        # Concatenate
        inter_cat = [torch.cat(v, dim=0) for v in inter_acc]
        if concat:
            embeddings = torch.cat(inter_cat, dim=-1)
        else:
            embeddings = inter_cat[0] if len(inter_cat) == 1 else inter_cat

        final_tensor = torch.cat(final_acc, dim=0) if return_final else None
        y_tensor = torch.cat(y_acc, dim=0) if (return_y and y_acc and len(y_acc) > 0) else None

        if return_final and return_y:
            return embeddings, final_tensor, y_tensor
        if return_final:
            return embeddings, final_tensor
        if return_y:
            return embeddings, y_tensor
        return embeddings


if __name__ == "__main__":
    import shutil
    from torch.utils.data import Dataset, DataLoader


    class TinyDataset(Dataset):
        def __init__(self, n=32, in_dim=10, n_classes=3, seed=0):
            g = torch.Generator().manual_seed(seed)
            self.x = torch.randn(n, in_dim, generator=g)
            self.y = torch.randint(0, n_classes, (n,), generator=g)

        def __len__(self):
            return self.x.size(0)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]


    class TinyNet(nn.Module):
        def __init__(self, in_dim=10, h=16, h2=12, out=3):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Linear(in_dim, h),  # layer name: seq.0
                nn.ReLU(),  # seq.1
                nn.Linear(h, h2),  # seq.2
                nn.ReLU(),  # seq.3
                nn.Linear(h2, out)  # seq.4
            )

        def forward(self, x):
            return self.seq(x)


    # --- Build model & data ---
    device = torch.device("cpu")
    model = TinyNet().to(device)
    ds = TinyDataset()
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    # --- Choose layers & modes (includes a 'both' to test expansion) ---
    target_layer_names = ["seq.2", "seq.0", "seq.0"]
    capture_modes = ["both", "output", "output"]
    entry_indices = [0] * len(target_layer_names)

    # Pre-compute expanded (layer,mode) list mirroring wrapper & cache ordering
    expanded_specs = []
    for l, m in zip(target_layer_names, capture_modes):
        if m == "both":
            expanded_specs.append((l, "input"))
            expanded_specs.append((l, "output"))
        else:
            expanded_specs.append((l, m))

    # --- Direct run (ground truth) ---
    direct_wrapper = ModelInputOutputWrapper(
        model,
        target_layer_names=target_layer_names,
        capture_modes=capture_modes,
        flatten=False,
        concat=False,
        entry_indices=entry_indices
    )

    direct_inter_acc = [[] for _ in expanded_specs]  # sized by expanded list
    direct_final_acc = []
    direct_y_acc = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            inter_list, final_out = direct_wrapper(x)
            assert len(inter_list) == len(expanded_specs), "Wrapper output length mismatch"
            for i, t in enumerate(inter_list):
                direct_inter_acc[i].append(t.detach().cpu())
            direct_final_acc.append(final_out.detach().cpu())
            direct_y_acc.append(y.detach().cpu())

    direct_inter = [torch.cat(v, dim=0) for v in direct_inter_acc]
    direct_final = torch.cat(direct_final_acc, dim=0)
    direct_y = torch.cat(direct_y_acc, dim=0)

    # --- Cache directory (fresh) ---
    cache_dir = os.path.join(os.path.dirname(__file__), "demo_cache")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)

    cache = LayerEmbeddingCache(
        model=model,
        dataloader=dl,
        cache_dir=cache_dir,
        device=device,
        # Add some reducers for testing get_available_reducers

    )

    # ---- Test get_available_reducers method ----
    print("\nTesting get_available_reducers method:")
    # seq.0 output dim is 16, seq.2 input dim is 16, seq.2 output dim is 12
    # Expected: pca(8) applies to 16 and 12. noop applies to all.
    available_s0_out = cache.get_available_reducers("seq.0", "output")

    # --- Request embeddings via cache ---
    cached_result = cache.get_embeddings(
        target_layer_names=target_layer_names,
        capture_modes=capture_modes,
        flatten=False,
        concat=False,
        entry_indices=entry_indices,
        return_y=True,
        return_final=True
    )
    cached_emb, cached_final, cached_y = cached_result

    # Normalize to list
    if isinstance(cached_emb, torch.Tensor):
        cached_list = [cached_emb]
    else:
        cached_list = cached_emb

    # Ensure expanded ordering matches
    assert len(cached_list) == len(expanded_specs), "Cache output length mismatch with expanded specs"

    # --- Assertions ---
    atol, rtol = 1e-6, 1e-5
    for i, (d, c, spec) in enumerate(zip(direct_inter, cached_list, expanded_specs)):
        assert d.shape == c.shape, f"Shape mismatch {spec}: {d.shape} vs {c.shape}"
        assert torch.allclose(d, c, atol=atol, rtol=rtol), f"Value mismatch in embedding {i} ({spec})"

    assert torch.allclose(direct_final, cached_final, atol=atol, rtol=rtol), "Final output mismatch"
    assert torch.equal(direct_y, cached_y), "Y mismatch"

    print("Embedding cache test passed (first retrieval).")
    for i, (t, spec) in enumerate(zip(cached_list, expanded_specs)):
        print(f"  cached[{i}] {spec} shape={t.shape}")
    print(f"  final shape={cached_final.shape}, y shape={cached_y.shape}")
    print(f"  cache dir: {cache_dir}")

    # ---- Second retrieval (should hit cache only, no file changes) ----
    seq0_out_path = os.path.join(cache_dir, "output", "seq.0.safetensors")
    assert os.path.isfile(seq0_out_path), "Expected cached file missing for second retrieval test."
    mtime_before = os.path.getmtime(seq0_out_path)

    second_result = cache.get_embeddings(
        target_layer_names=target_layer_names,
        capture_modes=capture_modes,
        flatten=False,
        concat=False,
        entry_indices=entry_indices,
        return_y=True,
        return_final=True
    )
    second_emb, second_final, second_y = second_result

    if isinstance(second_emb, torch.Tensor):
        second_list = [second_emb]
    else:
        second_list = second_emb

    assert len(second_list) == len(cached_list)
    for i, (a, b) in enumerate(zip(cached_list, second_list)):
        assert torch.allclose(a, b), f"Second retrieval mismatch at idx {i}"
    assert torch.allclose(cached_final, second_final), "Final output changed unexpectedly on second retrieval"
    assert torch.equal(cached_y, second_y), "Y changed unexpectedly on second retrieval"

    mtime_after = os.path.getmtime(seq0_out_path)
    assert mtime_before == mtime_after, "Cache file unexpectedly modified on read-only second retrieval"
    print("Second retrieval consistency test passed.")

    # ---- Direct wrapper via run_wrapper (matching API) ----
    direct_run = cache.run_wrapper(
        target_layer_names=target_layer_names,
        capture_modes=capture_modes,
        flatten=False,
        concat=False,
        entry_indices=entry_indices,
        return_y=True,
        return_final=True
    )
    run_emb, run_final, run_y = direct_run

    # Normalize for comparison
    if isinstance(run_emb, torch.Tensor):
        run_list = [run_emb]
    else:
        run_list = run_emb
    if isinstance(cached_emb, torch.Tensor):
        cached_list_cmp = [cached_emb]
    else:
        cached_list_cmp = cached_emb
    assert len(run_list) == len(cached_list_cmp), "run_wrapper embeddings length mismatch"

    for i, (a, b) in enumerate(zip(run_list, cached_list_cmp)):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), f"run_wrapper vs cache mismatch at {i}"
    assert torch.allclose(run_final, cached_final), "run_wrapper final mismatch"
    assert torch.equal(run_y, cached_y), "run_wrapper y mismatch"
    print("run_wrapper consistency test passed.")

    # ---- make_wrapper test (manual aggregation) ----
    wrapper_instance = cache.make_wrapper(
        target_layer_names=target_layer_names,
        capture_modes=capture_modes,
        flatten=False,
        concat=False,
        entry_indices=entry_indices,
        return_final=True,
        return_y=True
    )
    agg_emb_parts = None
    agg_final_parts = []
    agg_y_parts = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            emb_batch, final_batch, y_batch = wrapper_instance(xb, yb)
            if isinstance(emb_batch, torch.Tensor):
                emb_list = [emb_batch]
            else:
                emb_list = emb_batch
            if agg_emb_parts is None:
                agg_emb_parts = [[] for _ in emb_list]
            for i, t in enumerate(emb_list):
                agg_emb_parts[i].append(t.cpu())
            agg_final_parts.append(final_batch.cpu())
            agg_y_parts.append(y_batch.cpu())
    emb_from_wrapper = [torch.cat(v, dim=0) for v in agg_emb_parts]
    final_from_wrapper = torch.cat(agg_final_parts, dim=0)
    y_from_wrapper = torch.cat(agg_y_parts, dim=0)
    for i, (a, b) in enumerate(zip(emb_from_wrapper, cached_list_cmp)):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), f"make_wrapper mismatch at {i}"
    assert torch.allclose(final_from_wrapper, cached_final), "make_wrapper final mismatch"
    assert torch.equal(y_from_wrapper, cached_y), "make_wrapper y mismatch"
    print("make_wrapper consistency test passed.")

    # ---- Model mutation & rehash test (after all consistency checks) ----
    with torch.no_grad():
        model.seq[0].weight.add_(0.001)  # slight change
    assert cache.rehash() is False, "rehash() should detect model modification"
    print("Model modification detected by rehash().")

    try:
        cache2 = LayerEmbeddingCache(
            model=model,
            dataloader=dl,
            cache_dir=cache_dir,
            device=device
        )
        raise AssertionError("Expected RuntimeError due to model fingerprint mismatch, but none was raised.")
    except RuntimeError as e:
        print("Model fingerprint mismatch correctly raised on new initialization.")
        print(f"  Error: {str(e).splitlines()[0]}")

    print("All extended tests passed.")
