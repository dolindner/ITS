import threading
import types
from copy import deepcopy, copy
from typing import MutableMapping, Any, List
import hashlib
import time

import torch
from laplace import Laplace, LLLaplace
from laplace.utils import FeatureExtractor
from torch.nn.utils import vector_to_parameters

from confidence.model.base_model import ModelBasedConfidence


from embedding_cache import LayerEmbeddingCache as _LayerEmbeddingCacheHelper

# This is a very hacky way to ensure that we are not refitting laplace elements we have refitted before.
# It caches the result of the laplace fit and reuses if the paramters are the same.
_LAPLACE_FIT_CACHE: dict = {}  # key -> (laplace_obj, last_access_time)
_LAPLACE_CACHE_TIMEOUT = 120  # seconds
_LAPLACE_CACHE_SWEEP_INTERVAL = 60
_LAPLACE_CACHE_SWEEPER_THREAD = None
_LAPLACE_CACHE_SWEEPER_STOP = threading.Event()
_LAPLACE_CACHE_LOCK = threading.Lock()
#it caches and regularly cleanes them.


def _laplace_cache_sweeper():
    global _LAPLACE_CACHE_SWEEPER_THREAD
    while not _LAPLACE_CACHE_SWEEPER_STOP.is_set():
        time.sleep(_LAPLACE_CACHE_SWEEP_INTERVAL)
        now = time.time()
        with _LAPLACE_CACHE_LOCK:
            keys_to_del = [k for k, (_, t) in _LAPLACE_FIT_CACHE.items() if now - t > _LAPLACE_CACHE_TIMEOUT]
            for k in keys_to_del:
                # Get the laplace object before deletion
                laplace_obj, _ = _LAPLACE_FIT_CACHE[k]
                # Probably would have unintended side effects. TODO CHECK
                #if hasattr(laplace_obj, 'model'):
                #    for param in laplace_obj.model.parameters():
                #        param.grad = None

                # Delete the entry
                del _LAPLACE_FIT_CACHE[k]
                del laplace_obj
                print(f"[LaplaceCache] Removed expired cache entry: {k}")

            # Force garbage collection after cleanup
            if keys_to_del:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # If cache is empty, stop the thread
            if not _LAPLACE_FIT_CACHE:
                _LAPLACE_CACHE_SWEEPER_STOP.set()
                _LAPLACE_CACHE_SWEEPER_THREAD = None
                print("[LaplaceCache] Cache empty, sweeper thread exiting.")
                return


def _ensure_laplace_cache_sweeper():
    global _LAPLACE_CACHE_SWEEPER_THREAD
    if _LAPLACE_CACHE_SWEEPER_THREAD is None or not _LAPLACE_CACHE_SWEEPER_THREAD.is_alive():
        _LAPLACE_CACHE_SWEEPER_STOP.clear()
        t = threading.Thread(target=_laplace_cache_sweeper, daemon=True)
        t.start()
        _LAPLACE_CACHE_SWEEPER_THREAD = t


# for some reason the specific laplace method for last layer was missing this in the library so i inject it
# function to inject. Not sure why it was missing. Code based on laplace library.
def _nn_functional_samples(
        self,
        X: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
        n_samples: int = 100,
        generator: torch.Generator | None = None,
        **model_kwargs,
) -> torch.Tensor:

    fs = list()

    feats = None
    for sample in self.sample(n_samples, generator):
        vector_to_parameters(sample, self.model.last_layer.parameters())

        if feats is None:
            # Cache features at the first iteration
            f, feats = self.model.forward_with_features(
                X.to(self._device), **model_kwargs
            )
        else:
            # Used the cached features for the rest iterations
            f = self.model.last_layer(feats)

        fs.append(f.detach() if not self.enable_backprop else f)

    vector_to_parameters(self.mean, self.model.last_layer.parameters())
    fs = torch.stack(fs)

    return fs

#torch laplace only allows single output models, so we temporary cache the other outputs and get them later.
class OutputCacheWrapper(torch.nn.Module):
    """
    Wrapper that enables model that have multiple ouputs like intermediate features to work with torch-laplace.
    This wrapper causes non logit output the be saved which can later be gotten using get_cached_output.
    """
    def __init__(self, base_model:torch.nn.Module, index: int):
        super().__init__()
        self.base_model = base_model
        self.index = index
        self._cache: List[Any] = []
        self.call_count = 0
        self.output_was_tuple = False
        self.caching_enabled = False

    def enable_caching(self):
        """
        Enable or disable caching of outputs.
        """
        self.caching_enabled = True

    def disable_caching(self):
        """
        Disable caching of outputs and clear cache properly.
        """
        self.caching_enabled = False
        # Delete tensors explicitly before clearing list
        for cached_item in self._cache:
            if isinstance(cached_item, torch.Tensor):
                del cached_item
            elif isinstance(cached_item, (tuple, list)):
                for item in cached_item:
                    if isinstance(item, torch.Tensor):
                        del item
        self._cache.clear()
        self.call_count = 0


    def forward(self, x):
        outputs = self.base_model(x)
        if isinstance(outputs, torch.Tensor):
            raise ValueError(
                "The base model must return a tuple or list of outputs. "
                "If it returns a single tensor, wit should not have to be wrapped with this class."
            )

        # Convert to list for uniform caching
        indexed_outputs = outputs[self.index]
        if not self.caching_enabled:
            # If caching is disabled, return the indexed output directly
            return indexed_outputs

        if isinstance(outputs, tuple):
            # If the output is a tuple, take all elemetns but the indexed one
            outputs = outputs[:self.index] + outputs[self.index + 1:]
        else:
            outputs = outputs.pop(self.index)  # Remove the indexed output
        self._cache.append(outputs)
        self.call_count += 1
        return indexed_outputs

    def get_cached_output(self,stack=True,mean=True,stack_dim=0):
        if self.call_count == 0:
            raise ValueError("No outputs cached yet. Call forward() first.")
        elif self.call_count ==1:
            out = self._cache[0]
            self._cache.clear()
            self.call_count = 0
            return out

        #use _pytree to stack outputs
        if mean:
            out = torch.utils._pytree.tree_map(
                lambda *args: torch.stack(args, dim=stack_dim), *self._cache
            )
        else:
            out = copy(self._cache)

        if mean and stack:
            out = torch.utils._pytree.tree_map(
                lambda t: t.mean(dim = stack_dim), out
            )

        # Clear cache and explicitly delete references
        for cached_item in self._cache:
            if isinstance(cached_item, torch.Tensor):
                del cached_item


        #clear cache
        self._cache.clear()
        self.call_count = 0
        return out



#LLLaplace._nn_predictive_samples = LaplaceModelSamplingConfidence._nn_predictive_samples

from contextlib import contextmanager

@contextmanager
def temporarily_enable_grads(model: torch.nn.Module):
    """
    Temporarily sets all parameters' requires_grad=True, then restores their original state.

    Args:
        model: Module whose parameters' requires_grad will be temporarily enabled.

    Returns:
        Context manager yielding the same model.
    """
    requires_grad_backup = {p: p.requires_grad for p in model.parameters()}
    try:
        for p in model.parameters():
            p.requires_grad = True
        yield model
    finally:
        for p, req_grad in requires_grad_backup.items():
            p.requires_grad = req_grad


class LaplaceModelSamplingConfidence(ModelBasedConfidence):
    """
    Wraps torch-laplace Laplace and uses specified methods.
    """
    def __init__(
        self,
        base_model: torch.nn.Module,
        index = None,  # index of the output
        confidence=None,

        #sampling settings
        samples: int = 10, #only used for mc pred type
        pred_type: str = "glm",  # glm or nn for classification. nn only supports mc carle link aprx
        link_approx: str = "probit",  # 'bridge', 'bridge_norm', or 'mc' for classification In addition we supoprt none in which case we simply return the distribution.
        # probit approximates the expected sigmoid of the logits directly from the distribution. So the output is always a softmax. Bridge similary transform the gaussian to a dirichlet distribution.
        diagonal_output =False,  # diagonal glm output, only for 'glm' pred_type

        #return type settings
        index_laplace: int = None, # index for the laplace fitting will use the same as normal index if None


        softmax: bool = True, #only used for mc pred type, if true we apply softmax to the output, if false we return the raw logits.
        average: bool = True, #unly used for mc pred type, if true we average over samples, if false we return the samples directly.

        #laplace settings
        hessian_structure: str = "kron",# {'diag', 'kron', 'full', 'lowrank', 'gp'}
        subset_of_weights: str = "last_layer",

        # fit settings
        method="marglik",  # does not need a val loader "gridsearch" is the alternative
        kwargs_opt_prior: dict = None,  # see torch-laplace optimize_prior_precision for all kwargs.

        # keep the same
        subnetwork_indices = None,
        backend: str = None, #see BaseLaplace for details on backend

        # caching
        enable_fit_cache: bool = True,  # enable in-memory caching of fitted Laplace objects keyed by base model hash
        # laplace only adds paramaters that requires grad, so we need to unfreeze them.(This also must be done for loading from cache from state dict)
        set_all_parameters_trainable: bool = True,
        prior_precision: float = 1.0, #note not supported for cahcing
    ):
        """Initializes the LaplaceModelSamplingConfidence wrapper. Note no all setting combination fully implemtented.
        Recommend is to use mc sampling while returning the logits per sample and to let other modules handle them.

                Args:
                    base_model (torch.nn.Module): The underlying neural network module.
                    index (int, optional): Index of the target output tensor to cache and evaluate
                        confidence on. Defaults to None.
                    confidence (callable, optional): Confidence estimation function or module.
                        Defaults to None.
                    samples (int): Number of Monte Carlo samples to draw. Only relevant if
                        `link_approx="mc"`. Defaults to 10.
                    pred_type (str): The prediction type framework. Options are "glm"  or "nn". Note that "nn" only supports
                        Monte Carlo approximation. Defaults to "glm".
                    link_approx (str): The method used to approximate the link function/transform.
                        Options include "probit", "bridge", "bridge_norm", "mc", or "none" (returns
                        raw distribution properties). Defaults to "probit".
                    diagonal_output (bool): If True, uses a diagonal output covariance matrix. Only
                        applicable for "glm" `pred_type`. Defaults to False.
                    index_laplace (int, optional): The target index within the model output specifically
                        used for Laplace fitting. If None, mirrors `index`. Defaults to None.
                    softmax (bool): If True, applies a softmax transformation to Monte Carlo prediction
                        samples. If False, returns raw logits. Defaults to True.
                    average (bool): If True, computes the mean average across drawn Monte Carlo samples.
                        If False, samples are returned directly. Defaults to True.
                    hessian_structure (str): The structural approximation of the Hessian matrix.
                        Options include "diag", "kron", "full", "lowrank", or "gp". Defaults to "kron".
                    subset_of_weights (str): Determines which weights are treated probabilistically.
                        Options include "last_layer", "all", or "subnetwork". Defaults to "last_layer".
                    method (str): The optimization strategy for prior precision tuning. Typically
                        "marglik" (marginal likelihood) or "gridsearch". Defaults to "marglik".
                    kwargs_opt_prior (dict, optional): Additional configuration arguments passed directly
                        to the `optimize_prior_precision` routine in `torch-laplace`. Defaults to None.
                    subnetwork_indices (sequence, optional): Subnetwork indices required when
                        `subset_of_weights="subnetwork"`. Defaults to None.
                    backend (str, optional): The `torch-laplace` back-end engine configuration to handle
                        Hessian calculations. Defaults to None.
                    enable_fit_cache (bool): Enables RAM-based caching of fully fitted Laplace instances,
                        keyed uniquely by the model fingerprint. Defaults to True.
                    set_all_parameters_trainable (bool): Automatically enables gradients across model
                        parameters during the initialization and fitting stages. Defaults to True.
                    prior_precision (float): Initial precision parameter value for the Gaussian weight
                        prior.
                """
        super().__init__(base_model, confidence=confidence, index=index)
        print("prior precision at init:", prior_precision)
        # instantiate Laplace with same API as torch-laplace
        self.temperature = 1.0
        # Save a reference to the original base model for hashing
        self._base_model_for_hash = base_model
        self.enable_fit_cache = enable_fit_cache

        # Always use the robust fingerprint from embedding_cache
        self._base_model_init_hash = _LayerEmbeddingCacheHelper._model_fingerprint(base_model)

        self.index = index  # index of the output to cache, if None, no caching is done
        self.index_laplace = index_laplace or index  # index for the laplace fitting, if None, use the same as index

        if self.index_laplace is not None:
            # wrap the base model to cache the output at the index
            model = OutputCacheWrapper(base_model, self.index_laplace)
        else:
            model = base_model

        #set all parameters to require grad
        self.set_all_parameters_trainable = set_all_parameters_trainable

        if set_all_parameters_trainable:
            grad_context = temporarily_enable_grads(base_model)
        else:
            grad_context = nullcontext = contextmanager(lambda: (yield base_model))

        with grad_context:
            if subnetwork_indices is not None and subset_of_weights != "subnetwork":
                self.laplace = Laplace(
                    model,
                    likelihood="classification",
                    hessian_structure=hessian_structure,
                    prior_precision=prior_precision,
                    subset_of_weights=subset_of_weights,
                    subnetwork_indices=subnetwork_indices,
                    backend=backend,
                    enable_backprop=True,
                    temperature=self.temperature
                )
            else:
                self.laplace = Laplace(
                    model,
                    likelihood="classification",
                    hessian_structure=hessian_structure,
                    prior_precision=prior_precision,
                    subset_of_weights=subset_of_weights,
                    backend=backend,
                    enable_backprop=True,
                    temperature=self.temperature
                )
        #print precision after init
        print("prior precision after init:", self.laplace.prior_precision)

        #if type of laplace is Lllaplace inject function
        if isinstance(self.laplace, LLLaplace):
            print("Warning: Injecting _nn_predictive_samples into LLLaplace instance. This is a workaround for compatibility with torch-laplace. Not sure why it is not reimplemented even though parent implemnts it.")
            self.laplace._nn_functional_samples = types.MethodType(
                _nn_functional_samples, self.laplace
            ) #not working method of base class is still called

        self.pred_type = pred_type
        self.link_approx = link_approx
        self.samples = samples
        self.average = average
        self._fit_params = {}
        self.fit_other_kwargs = kwargs_opt_prior if kwargs_opt_prior is not None else {}
        self.diagonal_output = diagonal_output

        self.softmax = softmax  # only used for mc pred type, if true we apply softmax to the output, if false we return the raw logits.
        self.method = method
        self.experimental_last_layer_opt =False #TODO implement a way to cache intermediate values and only replace last layer weights if this is set to true.
        self.fitted = False
        self._cache_key = None
        self._last_cache_refresh = 0.0

    @property
    def model(self):
        """
        Return the base model, which is either the original model or the wrapped one.
        """
        if isinstance(self.laplace.model, OutputCacheWrapper):
            return self.laplace.model
        return self.laplace.model.model

    def _make_cache_key(self) -> str:
        # We reuse the hash from embedding cache.
        base_hash = self._base_model_init_hash

        # Make fit_other_kwargs deterministic for usage as a key
        if isinstance(self.fit_other_kwargs, dict):
            try:
                extras_items = tuple(sorted(self.fit_other_kwargs.items()))
            except Exception:
                extras_items = repr(self.fit_other_kwargs)
        else:
            extras_items = repr(self.fit_other_kwargs)
        #add extra attributes that (may) affect fit results.
        extras = "|".join(
            [
                str(getattr(self.laplace, "hessian_structure", "")),
                str(getattr(self.laplace, "subset_of_weights", "")),
                str(self.method),
                str(extras_items),
                str(self.pred_type),
                str(self.link_approx),
            ]
        )
        return f"{base_hash}|{extras}"
    @torch.no_grad()
    def fit(self, train_loader, validation_loader=None):
        """
        Fine-tune the Laplace posterior and optionally cache the fitted object.

        Very important method: ensures the Laplace posterior is fit and cached properly.

        Args:
            train_loader: DataLoader for training data used by laplace.fit.
            validation_loader: Optional DataLoader for validation used in prior optimization.

        Returns:
            None
        """
        if self.enable_fit_cache:
            _ensure_laplace_cache_sweeper()
            try:
                key = self._make_cache_key()
                self._cache_key = key  # Save for later refresh
                with _LAPLACE_CACHE_LOCK:
                    cached = _LAPLACE_FIT_CACHE.get(key, None)
                    if cached is not None:
                        print("Precision before loading from cache:", self.laplace.prior_precision)
                        precision_before = self.laplace.prior_precision
                        self.laplace = cached[0]
                        _LAPLACE_FIT_CACHE[key] = (self.laplace, time.time())
                        if isinstance(self.laplace, LLLaplace):
                            self.laplace._nn_functional_samples = types.MethodType(
                                _nn_functional_samples, self.laplace
                            )
                        self.fitted = True
                        print(f"Loaded fitted Laplace from in-memory cache (key={key})")
                        # Overwrite prior_precision if link_approx is "none"
                        if self.method == "none":
                            self.laplace.prior_precision = precision_before
                        print("Using cached Laplace fit, skipping re-fitting. Precision:", self.laplace.prior_precision)
                        return
            except Exception:
                pass

        if self.index_laplace is not None:
            self.model.disable_caching()
        with torch.enable_grad():
            if self.set_all_parameters_trainable:
                grad_context = temporarily_enable_grads(self._base_model_for_hash)
            else:
                grad_context = nullcontext = contextmanager(lambda: (yield self._base_model_for_hash))
            with grad_context:
                self.laplace.fit(train_loader)
                if self.method != "none":
                    self.laplace.optimize_prior_precision(
                        method=self.method, pred_type=self.pred_type, val_loader=validation_loader,
                        link_approx=self.link_approx, **self.fit_other_kwargs, init_prior_prec=self.laplace.prior_precision
                    )
            self.fitted = True


        if self.enable_fit_cache:
            key = self._make_cache_key()
            self._cache_key = key  # Save for later refresh
            with _LAPLACE_CACHE_LOCK:
                _LAPLACE_FIT_CACHE[key] = (self.laplace, time.time())
                _ensure_laplace_cache_sweeper()
            print(f"Stored fitted Laplace into in-memory cache (key={key})")





    def forward_no_link_approx(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Forward pass without link approximation. Always assumes glm pred_type.

        Args:
            x: Input tensor or batch for prediction.
            y: Optional labels for confidence computation.

        Returns:
            Tuple (confidence, model_output) or raises if not fitted/unsupported combination.
        """
        logits,logits_var = self.laplace._glm_predictive_distribution(x,diagonal_output=self.diagonal_output)
        tup = (logits,logits_var)


        if self.index_laplace is not None:
            output = self.model.get_cached_output(stack=False, mean=False, stack_dim=1)
            if isinstance(output, tuple):
                output = list(output)
                output.insert(self.index_laplace, tup)
                output = tuple(output)
            else:
                if isinstance(output, list):
                    output.insert(self.index_laplace, tup)
                else:
                    output[self.index_laplace] = tup
        else:
            output = tup

        confidence = self.confidence(output, y)
        if self.index is None:
            return confidence, output
        self.model.disable_caching()
        return confidence, output[self.index]

    def set_backprop(self, backprop: bool):
        """
        Set whether to backpropagate through the Laplace posterior.
        """
        self.laplace.enable_backprop = backprop
        if type(self.laplace.model) == FeatureExtractor:
            self.laplace.model.enable_backprop = backprop


    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Forward dispatcher to the chosen link approximation / prediction type.

        Very important method: verifies fit state, manages caching, and dispatches to the proper forward logic.

        Args:
            x: Input tensor or batch for prediction.
            y: Optional labels for confidence computation.

        Returns:
            Tuple (confidence, model_output) or raises if not fitted/unsupported combination.
        """
        # Refresh cache timeout only when forward is called, and only if enough time has passed
        if self.enable_fit_cache and self._cache_key is not None:
            now = time.time()
            if now - self._last_cache_refresh > 1.0:  # avoid excessive locking
                with _LAPLACE_CACHE_LOCK:
                    entry = _LAPLACE_FIT_CACHE.get(self._cache_key, None)
                    if entry is not None:
                        _LAPLACE_FIT_CACHE[self._cache_key] = (entry[0], now)
                self._last_cache_refresh = now

        if not self.fitted:
            raise RuntimeError("The Laplace model has not been fitted yet. Call .fit() before forward.")
        if self.index is not None:
            self.model.enable_caching()  # enable caching for the model
        if self.pred_type == "nn" and self.link_approx != "mc":
            self.model.disable_caching()  # disable caching for the model if not using mc
            raise ValueError(
                "For pred_type 'nn', link_approx must be 'mc'. Use forward_average for other link approximations."
            )
        if self.pred_type == "glm" and self.link_approx == "none":
            return self.forward_no_link_approx(x, y)

        if self.link_approx == "mc":
            return self.forward_monte_carlo(x, y)
        elif self.link_approx in ["probit", "bridge", "bridge_norm"] and self.pred_type == "glm":
            return self.forward_average(x, y)
        else:
            self.model.disable_caching()  # disable caching for the model if not using mc
            raise ValueError(
                f"Unsupported link approximation: {self.link_approx} for pred_type {self.pred_type}. "
                "Use 'mc' for NN or 'probit', 'bridge', 'bridge_norm' for GLM."
            )


    def forward_monte_carlo(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Forward pass with Monte Carlo sampling.

        Args:
            x: Input tensor for prediction.
            y: Optional labels for confidence computation.

        Returns:
            Tuple (confidence, output). Output shape depends on sampling/averaging settings.
        """
        if self.link_approx != "mc":
            raise ValueError(
                "For pred_type 'nn', link_approx must be 'mc'. Use forward_average for other link approximations."
            )

        # Ensure caching is enabled if needed
        if self.index_laplace is not None:
            self.model.enable_caching()

        # in this case simply call forward from laplace
        if self.softmax:
            mc_output = self.laplace.predictive_samples(
                x,
                pred_type=self.pred_type,
                diagonal_output=self.diagonal_output,
                n_samples=self.samples,
            )
        else:
            mc_output = self.laplace.functional_samples(
                x,
                pred_type=self.pred_type,
                diagonal_output=self.diagonal_output,
                n_samples=self.samples,
            )

        if self.average:
            mc_output = mc_output.mean(dim=0)
        else:
            mc_output = mc_output.permute(1, 0, *range(2, mc_output.dim()))

        if self.index_laplace is not None:
            output = self.model.get_cached_output(stack=True, mean=self.average, stack_dim=1)
            if isinstance(output, tuple):
                output = list(output)
                output.insert(self.index_laplace, mc_output)
                output = tuple(output)
            else:
                if isinstance(output, list):
                    output.insert(self.index_laplace, mc_output)
                else:
                    output[self.index_laplace] = mc_output
        else:
            output = mc_output

        confidence = self.confidence(output, y)
        # always average output for output
        if not self.average:
            # average here for output
            output = output.mean(dim=1)

        if self.index is None:
            return confidence, output

        self.model.disable_caching()
        return confidence, output[self.index]

    def forward_average(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Forward pass with averaging over approximate link transform samples.

        Args:
            x: Input tensor for prediction.
            y: Optional labels for confidence computation.

        Returns:
            Tuple (confidence, outputs)
        """
        # in this case simply call forward from laplace
        if not self.softmax:
            raise ValueError(
                "Non Softmax output is only supported for Monte Carlo sampling. "
            )

        average_output = self.laplace(x, pred_type=self.pred_type, link_approx=self.link_approx, n_samples=self.samples,
                                     diagonal_output=self.diagonal_output)

        if self.index_laplace is not None:
            outputs = self.model.get_cached_output(
                stack=False, mean=False, stack_dim=1
            )
            if isinstance(outputs, tuple):
                outputs = list(outputs)
                outputs.insert(self.index_laplace, average_output)
                outputs = tuple(outputs)
            else:
                if isinstance(outputs, list):
                    outputs.insert(self.index_laplace, average_output)
                else:
                    outputs[self.index_laplace] = average_output
        else:
            outputs = average_output



        confidence = self.confidence(outputs, y)

        if self.index is None:
            return confidence, outputs
        self.model.disable_caching()
        return confidence, outputs[self.index]

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    class DualOutputModule(torch.nn.Module):
        def __init__(self, in_features: int, hidden: int, num_classes: int):
            super().__init__()
            self.shared = torch.nn.Sequential(
                torch.nn.Linear(in_features, hidden),
                torch.nn.ReLU(),
            )
            self.classifier = torch.nn.Linear(hidden, num_classes)
            self.regressor = torch.nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor):
            h = self.shared(x)
            logits = self.classifier(h)
            return logits, h


    X = torch.randn(5000, 64)
    y = torch.randint(0, 10, (5000,))
    train_loader = DataLoader(TensorDataset(X, y), batch_size=128)

    # use a larger model
    base_model = DualOutputModule(in_features=64, hidden=512, num_classes=10)
    conf = LaplaceModelSamplingConfidence(base_model, index=0, enable_fit_cache=True)

    # first fit (should perform full fit)
    t0 = time.time()
    conf.fit(train_loader)
    t1 = time.time() - t0
    print(f"First fit time: {t1:.3f}s")
    time.sleep(0.1)  # wait a bit to ensure cache sweeper can run if needed

    # second fit (should hit cache and be fast)
    t0 = time.time()
    conf.fit(train_loader)
    t2 = time.time() - t0
    print(f"Second fit (cached) time: {t2:.6f}s")

    conf.confidence = lambda out, y=None: out[0][0].softmax(-1).max(-1).values