"""
Contains slightly changed version of its module that can also accept confidence modules.
 If it is none it falls back to original implementation.
Original: https://github.com/johschm/its
Changes include adding confidence module support. Adding simple reshaping for non images.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from its.transform import orbit_sampling, identity


# taken from its package and adapted to handle non images as well as allowing a settable confidence module.

def entropy(p):
    """
    Computes the entropy of the given probability.

    Parameters:
    p (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: The entropy.
    """
    return - torch.sum(p * torch.log2(p) + 1e-10, dim=-1)


def curvature(s):
    """
    Computes the curvature of a given tensor by calculating the second-order derivative.

    Parameters:
    s (torch.Tensor): The input tensor for which the curvature is to be computed.

    Returns:
    torch.Tensor: The curvature of the input tensor, which is the second-order derivative of the input tensor.
    """

    # Calculate the first-order derivative (gradient)
    # edge_order=2 to handle endpoint gradients
    g = torch.gradient(s, edge_order=1, dim=-1)[0]
    # Compute the curvature (second-orde derivative) of the gradient
    return torch.gradient(g, edge_order=1, dim=-1)[0]


def highlight_subplot(ax, color='red', line_width=8):
    """
    Highlights a subplot by adding a colored rectangle at the top of the subplot.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object of the subplot to be highlighted.
    color (str, optional): The color of the rectangle. Default is 'red'.
    line_width (int, optional): The width of the rectangle's line. Default is 8.

    Returns:
    None
    """
    ax_coordinates = ax.axis()
    rec = Rectangle(
        (ax_coordinates[0], ax_coordinates[2]),  # xy
        ax_coordinates[1],  # width
        1,  # height
        fill=True, lw=line_width, color=color)
    rec = ax.add_patch(rec)
    rec.set_clip_on(False)


def gaussian_filter1d(input_tensor, sigma, radius, mode='replicate'):
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    gaussian = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = gaussian / gaussian.sum()
    # out_channels x in_channels x kernel_size
    kernel = kernel.view(1, 1, -1).tile((input_tensor.shape[1], input_tensor.shape[1], 1))
    padded = F.pad(input_tensor, (radius, radius), mode=mode)
    filtered = F.conv1d(padded, kernel.to(input_tensor.device))
    return filtered


def gaussian_filter1d_channel_wise(input_tensor, sigma, radius, mode='replicate'):
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    gaussian = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = gaussian / gaussian.sum()
    kernel = kernel.view(1, 1, -1).tile((input_tensor.shape[1], 1, 1))

    padded = F.pad(input_tensor, (radius, radius), mode=mode)
    filtered = F.conv1d(
        padded,
        kernel.to(input_tensor.device),
        groups=input_tensor.shape[1]
    )
    return filtered


class InverseTransformationSearch:
    """
    This is the ITS inference module, which you simply insert
    as a pre-processing step in front of classifier during inference.
    """

    def __init__(self,
                 model,
                 transformation,
                 domain,
                 n_samples=17,
                 n_hypotheses=1,
                 mc_steps=1,
                 change_of_mind='score',
                 en_unique_class_condition=False,
                 labels=None,
                 line_thickness=1,
                 fontsize=16,
                 gaussian_filter_channel_wise=False,
                 confidence_module=None,
                 extend=1,
                 # for our metrics we do not need extra padding only for original as it calculated differences and gaussian smoothing. So for those this should stay 1.
                 resampling_method=None,
                 ):
        """
        Initialises the ITS inference module.

        Parameters:
        n_hypotheses (int>0): The number of hypotheses to be considered in the search.
                              If `en_unique_class_condition=True` this is bounded to max `n_hypotheses=n_classes`.
        change_of_mind (str): The method for changing of mind during the search.
                              This can be either:
                                - `score`: This uses the evaluation scores.
                                           This is a greedy pick of the highest scored result over the entire search.
                                - `class`: This uses the class counts per hypothesis branch. The highest support wins.
                                - `off`: Disable change of mind and use the initial best hypothesis.

        Returns:
        None
        """

        assert change_of_mind in ['score', 'class', 'off'], "Choose either of 'score', 'class', 'off'."

        # You can adjust these settings without re-instantiating the class
        # e.g. by changing `its.n_hypotheses=3` and then run `its.infer(x)`
        self.change_of_mind = change_of_mind
        self.model = model
        self.transformation = transformation
        self.n_samples = n_samples
        self.domain = domain
        self.transformation_schedule = transformation
        self.en_unique_class_condition = en_unique_class_condition
        self.n_hypotheses = n_hypotheses
        self.mc_steps = mc_steps

        # parameters initialised in `infer`
        self.batch_size = None
        self.max_batch_size_override = None
        self.T = None
        self.history = None

        # parameters for plotting
        self.en_plot = False
        self.line_thickness = line_thickness
        self.fontsize = fontsize
        self.labels = np.arange(100) if labels is None else labels
        self.gaussian_filter_channel_wise = gaussian_filter_channel_wise
        self.confidence_module = confidence_module

        self.extend = extend
        self.resampling_method = resampling_method
        # recorded once on first infer: original input ndim (e.g. 4 for images B,C,H,W; 3 for sequences B,L,F)
        self._orig_input_dim = None
        self._matrix_dim = None
        self.debug_energy_only = False

    @torch.no_grad()
    def infer(self, x, plot_idx=None, return_best: bool = False, matrix_dim: int | None = None,
              max_batch_size_override: int | None = None):
        """
        This is the inference function which initiates the search process.

        Parameters:
        x (torch.Tensor): The input image batch. (batch x channel x width x height)
        plot_idx (bool, optional): The index of the batch for which to plot the search results
                                   Default is None, which means no plotting is performed.
        return_best (bool, optional): If True, return (best_param, best_error, best_logits)
                                      where best_error = -confidence (primal score) for chosen hypotheses.
                                      If False (default), return (x_t, z) as before.
        matrix_dim (int, optional): The dimension of the transformation matrix (e.g., 3 for 2D, 4 for 3D).
                                    If not provided, it will be inferred.
        max_batch_size_override (int, optional): If provided, overrides batch size for model/confidence calls
                                                 to prevent OOM errors.

        Returns:
        x_t (torch.Tensor) or best_param (torch.Tensor) depending on return_best.
        z (torch.Tensor) or best_error, best_logits
        """
        # remember original input dimensionality on first call
        if self._orig_input_dim is None:
            self._orig_input_dim = x.dim()
        self.batch_size = int(x.shape[0])
        self.max_batch_size_override = max_batch_size_override
        if matrix_dim is not None:
            self._matrix_dim = matrix_dim
        else:
            self._matrix_dim = 3  # Fallback for old transformations
        self.T = identity((self.batch_size, self.n_hypotheses), d=self._matrix_dim).to(x.device)
        self.history = {"orbit": [], "embedding": [], "score": [], "n_max": [], "T": []}
        self.en_plot = plot_idx is not None

        # pad x dims such that they are 4 dims (B,1,1,D) if input is 1d or 3 dims (B,1,H,W) if input is 2d
        if self._orig_input_dim not in [4, None]:
            if self._orig_input_dim == 3:
                x = x[:, None, ...]
            elif self._orig_input_dim == 2:
                x = x[:, None, None, ...]
            else:
                raise ValueError(f"Input with {self._orig_input_dim} dims is not supported. "
                                 f"Only 2D (B,L) or 3D (B,C,H,W) inputs or 4D (B,C,H,W) inputs are supported.")

        # ensure consistent shapes
        x = torch.tile(x[:, None, ...], [1, self.n_hypotheses, 1, 1, 1])
        # main search loop: iterate over all possible transformations
        scores = None
        for i in range(len(self.transformation_schedule)):
            # keep the same x as input but the transformation matrix is composed during search
            x_t, z, scores = self._breadth_first_search_step(x, level=i)

        # plot the gathered search results
        if self.en_plot:
            self._plot_tree(batch_index=plot_idx)

        # hypothesis testing: change of mind
        if self.change_of_mind != 'off':
            # request indices from _change_of_mind so we can reorder T and scores as well
            x_t, z, indices = self._change_of_mind(x_t, z, scores=scores, return_indices=True)
        else:
            # identity indices shape: (batch x n_hypotheses)
            indices = torch.arange(self.n_hypotheses, device=x.device)[None, :].expand(self.batch_size, -1)

        # remove potential extra dim from x_t if input was not image
        if self._orig_input_dim not in [4, None]:
            x_t = x_t.flatten(start_dim=1, end_dim=x_t.dim() - self._orig_input_dim)

        if return_best:
            # best_param: reorder self.T (batch x n_hypotheses x 3 x 3) according to indices
            idx_for_T = indices[..., None, None].expand(-1, -1, self.T.shape[-2], self.T.shape[-1])
            best_param = torch.gather(self.T, dim=1, index=idx_for_T)
            # best_error: negative of the last primal confidence scores (scores) reordered
            # scores is (batch x n_hypotheses)
            best_error = -torch.gather(scores, dim=1, index=indices)
            # best_logits: z is already reordered by change_of_mind
            best_logits = z
            return best_param, best_error, best_logits, x_t

        return x_t, z

    def _change_of_mind(self, x_t, z, scores=None, return_indices: bool = False):
        """
        In each level of the search, the identity transform can be selected as candidate.
        Therefore, only the final candidate list must be reordered by the change of mind logic.

        Args:
            x_t: The hypotheses list canonical form of the input.
            z: The hypotheses list of output logit scores of the model.
            scores (torch.Tensor, optional): Evaluation scores of each transformation. Required for `score` mode.
            return_indices (bool): If True, return the indices used to reorder hypotheses as third value.

        Returns:
            x_t, z (and indices if return_indices=True)
        """
        if self.change_of_mind == 'score':
            # compute accumulated score value over all levels
            # (n_levels x batch x n_hypotheses x n_samples) -> (batch x n_hypotheses)
            scores_accum = torch.from_numpy(np.stack(self.history["score"])).max(-1).values.sum(0)
            indices = torch.sort(scores_accum, dim=-1, descending=True).indices.to(x_t.device)
        else:
            raise NotImplementedError(f"The change of mind mode {self.change_of_mind} is currently not implemented. "
                                      f"We refractored our prototypical code for the sake of simplicity "
                                      f"and a cleaner code base. The `class`-based change of mind mechanism requires "
                                      f"a stored class list over the search. Due to the minor differences "
                                      f"(see our paper), we see no need for it in this implementation.")

        x_t = torch.gather(x_t, dim=1, index=torch.tile(indices[..., None, None, None], [1, 1] + list(x_t.shape[-3:])))
        z = torch.gather(z, dim=1, index=torch.tile(indices[..., None], [1, 1, z.shape[-1]]))
        if return_indices:
            return x_t, z, indices
        return x_t, z

    @staticmethod
    def _estimate_confidence(z, channel_wise=False):
        """
        Estimates the confidence score of the input `z`.

        Args:
            z: The output logit scores of the model (batch x n_hypotheses x n_classes)

        Returns:
            confidence: The estimated confidence of the input `z` (batch x n_hypotheses)
        """
        energies = torch.log(torch.exp(z).sum(dim=-1))
        n_hypotheses = energies.shape[1]
        if channel_wise:
            # apply gaussian filter to each channel(hypothesis) separately
            energies = gaussian_filter1d_channel_wise(energies, sigma=2, radius=3, mode='replicate') * n_hypotheses
        else:
            energies = gaussian_filter1d(energies, sigma=2, radius=3, mode='replicate')
        return -curvature(energies.clone().detach().to(device=z.device))
        # todo add energy option -torch.tensor(energies[..., 1:self.n_samples+1], device=z.device)

    def _predict(self, x):
        """
        Calls `self.model` with the input `x`.
        If `mc_samples` == 1: Then the predictions are deterministic.
        Else : The predictions are stochastic.

        Help: For Huggingface or Torch models, you can access the logits by
        resnet logits model.fn; vgg logits model.classifier[6] or similar.

        Args:
            x (torch.Tensor): The input image batch (batch * n_hypotheses * samples x n_channels x height x width)

        Returns:
            logits (torch.Tensor): The output logit scores. (batch * n_hypotheses * samples x n_classes)
        """
        if self.max_batch_size_override and x.shape[0] > self.max_batch_size_override:
            outputs = []
            for i in range(0, x.shape[0], self.max_batch_size_override):
                batch_x = x[i:i + self.max_batch_size_override]
                outputs.append(self._predict_single_batch(batch_x))
            return torch.cat(outputs, dim=0)
        return self._predict_single_batch(x)

    def _predict_single_batch(self, x):
        """Helper for _predict to run a single batch through the model."""
        if self.mc_steps == 1:
            self.model.eval()
            return self.model(x)
        self.model.train()
        predictions = []
        for i in range(self.mc_steps):
            with torch.no_grad():
                predictions.append(self.model(x))
        return torch.stack(predictions).mean(dim=0)

    def _predict_confidence(self, x):
        """
        Calls `self.confidence_module` with input `x`, handling batching.
        If the confidence module returns no logits, it falls back to `self._predict`.
        """
        if self.max_batch_size_override and x.shape[0] > self.max_batch_size_override:
            z_was_None = False
            scores, zs = [], []
            for i in range(0, x.shape[0], self.max_batch_size_override):
                batch_x = x[i:i + self.max_batch_size_override]
                score_chunk, z_chunk, z_was_None_b = self._predict_confidence_single_batch(batch_x)
                scores.append(score_chunk)
                zs.append(z_chunk)
                z_was_None = z_was_None or z_was_None_b
            score = torch.cat(scores, dim=0)
            z = torch.cat(zs, dim=0)
            return score, z, z_was_None
        return self._predict_confidence_single_batch(x)

    def _predict_confidence_single_batch(self, x):
        """Helper for _predict_confidence to run a single batch."""
        score, z = self.confidence_module(x)
        z_was_None = False
        if z is None or z.numel() == 0:
            z = self._predict(x)
            z_was_None = True
            if not hasattr(self, "_warned_confidence_module_no_logits"):
                print("Warning: confidence_module returned no logits, falling back to _predict for logits.")
                self._warned_confidence_module_no_logits = True
        return score, z, z_was_None

    def _evaluate_orbit(self, x, level):
        """
        Evaluates the orbit at the current level by
            (i) Obtaining regular samples along orbit and
            (ii) Evaluating the confidence of the samples.

        Args:
            x (torch.Tensor): The input image batch (batch x n_hypotheses x samples x n_channels x height x width)
            level (int): The current level of the search

        Returns:
            orbit (torch.Tensor): The regular samples. (batch x n_hypotheses x n_samples x n_channels x height x width)
            scores (torch.Tensor): Evaluation scores of the current orbit. (batch x n_hypotheses x samples)
            z (torch.Tensor): Evaluation scores of the current orbit. (batch x n_hypotheses x n_classes)
            T (torch.Tensor): The transformation matrix. (batch x n_hypotheses x n_samples x 3 x 3)
        """
        # Optimization for level 0: compute orbit for one hypothesis and expand
        OPT = True
        if level == 0 and OPT:
            n_hypotheses_orig = self.n_hypotheses
            self.n_hypotheses = 1
            x_in = x[:, 0:1]
            # this line is correct
            T_prior = self.T[[0]].flatten(end_dim=1)[:, None].expand((-1, self.n_samples + 2 * self.extend, -1, -1))

        else:
            n_hypotheses_orig = self.n_hypotheses
            x_in = x
            T_prior = self.T.flatten(end_dim=1)[:, None].expand((-1, self.n_samples + 2 * self.extend, -1, -1))

        # (batch x n_hypotheses x n_samples x 3 x 3) -> (batch * n_hypotheses x n_samples+2 x 3 x 3)

        # Obtain regular samples along orbit (batch x n_hypotheses x n_samples+2 x n_channels x height x width)
        # as well as the corresponding transformation matrices (batch x n_hypotheses x n_samples x 3 x 3)
        orbit, T = orbit_sampling(x_in.flatten(end_dim=1),
                                  self.transformation_schedule[level],  # apply the transformation of that orbit
                                  n_samples=self.n_samples,  # number of regular samples
                                  domain=[self.domain[level], ],  # the domain of the current level
                                  T=T_prior, extend=self.extend, resample_fn=self.resampling_method,
                                  orig_data_dim=self._orig_input_dim, matrix_dim=self._matrix_dim)
        # obtain the representation of the input (batch * n_hypotheses * n_samples+2 x embedding_dim)

        z_was_none = False
        if self.confidence_module is not None:
            with torch.no_grad():
                if self._orig_input_dim is None or self._orig_input_dim == 4:
                    # original image path (unchanged)
                    flat_in = orbit.flatten(end_dim=2)
                    score, z, z_was_none = self._predict_confidence(flat_in)
                else:
                    # non-image path: collapse batch/hypotheses into leading dim, then merge trailing dims
                    flat_in = orbit.flatten(start_dim=0, end_dim=2)
                    flat_in = flat_in.flatten(end_dim=flat_in.dim() - self._orig_input_dim)
                    score, z, z_was_none = self._predict_confidence(flat_in)
            orbit = orbit.unflatten(0, (self.batch_size, self.n_hypotheses)).squeeze(dim=2)

            # the number of hypothesis dimension has images of different batches if non image data(maybe image data now as well have not checked yet)
            # if orbit is not dim 6 as for image data readd extra dimensions after 3 batch dims
            if orbit.dim() == 5 and self._orig_input_dim not in [4, None]:
                orbit = orbit[:, :, :, None, :, :]
            if orbit.dim() == 4 and self._orig_input_dim not in [4, None]:
                orbit = orbit[:, :, :, None, None, :]

            z = z.unflatten(0, (self.batch_size, self.n_hypotheses, self.n_samples + 2 * self.extend))
            score = score.unflatten(0, (self.batch_size, self.n_hypotheses, self.n_samples + 2 * self.extend))
            # squeeze away single orbit dim as len(T) = 1
            # T is a list of length 1
            T = T[0].unflatten(0, (self.batch_size, self.n_hypotheses))
            # estimate the confidence scores (batch x n_hypotheses x n_samples)
        else:
            if self._orig_input_dim is None or self._orig_input_dim == 4:
                z = self._predict(orbit.flatten(end_dim=2))
            else:
                flat_in = orbit.flatten(start_dim=0, end_dim=2)
                flat_in = flat_in.flatten(end_dim=flat_in.dim() - self._orig_input_dim)
                z = self._predict(flat_in)
            z = z.unflatten(0, (self.batch_size, self.n_hypotheses, self.n_samples + 2 * self.extend))
            # squeeze away single orbit dim as len(T) = 1
            orbit = orbit.unflatten(0, (self.batch_size, self.n_hypotheses)).squeeze(dim=2)

            if orbit.dim() == 5 and self._orig_input_dim not in [4, None]:
                orbit = orbit[:, :, :, None, :, :]
            if orbit.dim() == 4 and self._orig_input_dim not in [4, None]:
                orbit = orbit[:, :, :, None, None, :]

            # T is a list of length 1
            T = T[0].unflatten(0, (self.batch_size, self.n_hypotheses))
            if not self.debug_energy_only:
                # estimate the confidence scores (batch x n_hypotheses x n_samples)
                score = self._estimate_confidence(z, self.gaussian_filter_channel_wise)
            else:
                # use energy as score for debugging purposes
                score = torch.log(torch.exp(z).sum(dim=-1))

        # Restore n_hypotheses if changed for level 0 optimization
        if level == 0 and OPT:
            self.n_hypotheses = n_hypotheses_orig
            orbit = orbit.expand(-1, self.n_hypotheses, -1, -1, -1, -1)
            score = score.expand(-1, self.n_hypotheses, -1)
            z = z.expand(-1, self.n_hypotheses, -1, -1)
            T = T.expand(-1, self.n_hypotheses, -1, -1, -1)

        # begin injected fix attempt not tested
        expected_len = self.n_samples + 2 * self.extend
        cur_len = orbit.shape[2]
        if cur_len < expected_len:
            missing = expected_len - cur_len
            # distribute padding: first try to satisfy both sides up to `extend`
            front_pad = min(self.extend, missing // 2)
            back_pad = min(self.extend, missing - front_pad)
            # if still missing (missing > 2*extend or odd distribution), add remainder to the back
            remainder = missing - front_pad - back_pad
            back_pad += remainder

            if front_pad > 0:
                orbit_front = orbit[:, :, 0:1, ...].expand(-1, -1, front_pad, -1, -1, -1)
                z_front = z[:, :, 0:1, :].expand(-1, -1, front_pad, -1)
                score_front = score[:, :, 0:1].expand(-1, -1, front_pad)
                T_front = T[:, :, 0:1, ...].expand(-1, -1, front_pad, -1, -1)
                orbit = torch.cat([orbit_front, orbit], dim=2)
                z = torch.cat([z_front, z], dim=2)
                score = torch.cat([score_front, score], dim=2)
                T = torch.cat([T_front, T], dim=2)

            if back_pad > 0:
                orbit_back = orbit[:, :, -1:, ...].expand(-1, -1, back_pad, -1, -1, -1)
                z_back = z[:, :, -1:, :].expand(-1, -1, back_pad, -1)
                score_back = score[:, :, -1:].expand(-1, -1, back_pad)
                T_back = T[:, :, -1:, ...].expand(-1, -1, back_pad, -1, -1)
                orbit = torch.cat([orbit, orbit_back], dim=2)
                z = torch.cat([z, z_back], dim=2)
                score = torch.cat([score, score_back], dim=2)
                T = torch.cat([T, T_back], dim=2)
        # end injected fix attempt

        return (
            orbit[:, :, self.extend:self.n_samples + self.extend],
            score[:, :, self.extend:self.n_samples + self.extend],
            z[:, :, self.extend:self.n_samples + self.extend],
            T[:, :, self.extend:self.n_samples + self.extend]
        ), z_was_none

    def select_candidates(self, score, pred_class, level, z_was_none=False):
        """
        Selects the top-k candidates for each hypothesis based on their scores.
        If `en_unique_class_condition` is True, the selection is done based on the
        maximum score for each hypothesis across all classes. Otherwise, the selection
        is done based on the maximum score for each hypothesis on the current level.

        Args:
            score: The evaluation scores of the current orbit. (batch x n_hypotheses x n_samples)
            pred_class: The predicted classes of the input samples. (batch x n_hypotheses x n_samples)
            level: The current level of the search
            z_was_none (bool): If True, pred_class is not reliable.

        Returns:
            selected_candidates: The indices of the selected candidates. (batch x n_hypotheses)
        """
        if not hasattr(self, "_warned_n_hypotheses"):
            self._warned_n_hypotheses = False

        if not self.en_unique_class_condition:
            if level == 0:
                available = score[:, 0].shape[-1]
                k = min(self.n_hypotheses, available)
                if self.n_hypotheses > available and not self._warned_n_hypotheses:
                    print(
                        f"Warning: n_hypotheses ({self.n_hypotheses}) > available candidates ({available}), reducing k to {k}")
                    self._warned_n_hypotheses = True
                return torch.topk(score[:, 0], k=k, dim=-1).indices
            else:
                return torch.argmax(score, dim=-1)

        if level == 0:
            score, pred_class = score[:, 0], pred_class[:, 0]
        n_max = torch.zeros((self.batch_size, self.n_hypotheses), device=score.device, dtype=torch.int64)
        for i in range(self.n_hypotheses):
            if level == 0:
                index = torch.argmax(score, dim=-1)
                c = pred_class[torch.arange(score.shape[0]), index]
                score = torch.where(pred_class == c[:, None], torch.full_like(score, -torch.inf), score)
                n_max[:, i] = index
            else:
                index = torch.argmax(score[:, i], dim=-1)
                c = torch.gather(pred_class[:, i], -1, index[:, None])[:, None]
                score = torch.where(pred_class == c, torch.full_like(score, -torch.inf), score)
                n_max[:, i] = index
        return n_max

    def _breadth_first_search_step(self, x, level: int):
        """ Performs one breadth first search step.

        Args:
            x (torch.Tensor): The input to this search step (batch x channel x height x width).
            level (int): The current level of the search.

        Returns:
            x_incumbent (torch.Tensor): The best performing input transformation.
            z_incumbent (torch.Tensor): The corresponding embedding.
            s_primal (torch.Tensor): The corresponding score.
        """
        # compute the highest class scores and orbits
        (orbit, s, z, T), z_was_none = self._evaluate_orbit(x, level)
        # select the k best samples from the orbit -> (batch x n_hypotheses)
        n_max = self.select_candidates(s, torch.argmax(z, dim=-1), level, z_was_none=z_was_none)
        # update incumbent (input): (batch x n_hypotheses x n_samples x channel x height x width)
        x_incumbent = torch.gather(orbit, dim=2, index=torch.tile(
            n_max[..., None, None, None, None], [1, 1, 1] + list(orbit.shape[-3:]))).squeeze(2)
        # update the accumulative transformation matrix: (batch x n_hypotheses x n_samples x 3 x 3)
        self.T = torch.gather(T, dim=2, index=torch.tile(
            n_max[..., None, None, None], [1, 1, 1, self._matrix_dim, self._matrix_dim])).squeeze(2)
        # update incumbent (embedding): (batch x n_hypotheses x n_samples x n_classes)
        z_incumbent = torch.gather(z, dim=2, index=torch.tile(
            n_max[..., None, None], [1, 1, 1, z.shape[-1]])).squeeze(2)
        # update primal (confidence score): (batch x n_hypotheses x n_samples)
        s_primal = torch.gather(s, dim=2, index=n_max[..., None]).squeeze(2)
        # store the results
        self.history["orbit"].append(orbit.detach().cpu().numpy())
        self.history["embedding"].append(z.detach().cpu().numpy())
        self.history["score"].append(s.detach().cpu().numpy())
        self.history["n_max"].append(n_max.detach().cpu().numpy())
        self.history["T"].append(T.detach().cpu().numpy())

        return x_incumbent, z_incumbent, s_primal

    def _plot_tree(self, batch_index=0):
        """
        Plots the tree for a given batch index.

        Args:
            batch_index: The index of the batch to plot.

        Returns:
            Returns the plot as a matplotlib figure.

        """
        fig = plt.figure(layout='constrained',
                         figsize=(self.n_samples * 1.5, self.n_hypotheses * 2 * len(self.transformation)))
        subfigs = fig.subfigures(len(self.transformation), 1, wspace=0.07)
        subfigs = subfigs if len(self.transformation) > 1 else [subfigs, ]

        # normalize scores globally over the entire tree
        # scores: level x (B x K x O)
        stacked_tensor = np.stack(self.history["score"]).transpose((1, 0, 2, 3))  # bring batch up front
        min_val = stacked_tensor.reshape(self.batch_size, -1).min(axis=-1)[:, None, None]
        max_val = stacked_tensor.reshape(self.batch_size, -1).max(axis=-1)[:, None, None]

        for i in range(len(self.transformation)):
            scores = (self.history["score"][i] - min_val) / (max_val - min_val)
            self._plot_level(level=i, batch_index=batch_index, fig=subfigs[i],
                             orbit_samples=self.history["orbit"][i],
                             embedding=self.history["embedding"][i],
                             candidates=self.history["n_max"][i],
                             scores=scores)
        return fig

    def _plot_level(self, fig, orbit_samples: torch.tensor, embedding: torch.tensor, scores: torch.tensor,
                    candidates: torch.tensor, level: int, batch_index=0):
        """
        Plots on level of the search tree.

        Args:
            fig: The figure to plot on.
            orbit_samples: The samples of the orbit.
            embedding: The embedding.
            scores: The scores.
            candidates: The candidates.
            level: The level of the tree.
            batch_index: The batch index.

        Returns:
            None.
        """
        ax = fig.subplots(self.n_hypotheses if level != 0 else 1, self.n_samples)
        ax = ax if level != 0 else ax[None]

        # map scores to colors
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', '#176fc1'], N=256)
        colors = cmap(scores)

        # loop over branch stacks
        for k in range(self.n_hypotheses if level != 0 else 1):
            label_idx = np.argmax(embedding[batch_index, k], axis=-1)

            # loop over orbit samples (elements in a branch stack)
            for s in range(self.n_samples):

                # extend the image by a box above
                im = orbit_samples[batch_index, k, s].transpose((1, 2, 0))
                im = np.tile(im, (1, 1, 3)) if im.shape[-1] == 1 else im

                # some pre-processors do not map to [0,1]
                if im.min() != 0.:
                    im = (im - im.min()) / (im.max() - im.min())

                # add score-colored rectangle
                color_rec = np.tile(np.array(colors[batch_index, k, s, :3])[None, None],
                                    (5 * self.line_thickness, im.shape[1], 1))
                im = np.concatenate([color_rec, im], axis=0)

                # plot the image
                ax[k, s].imshow(im, cmap='gray')
                ax[k, s].set_xlim(0, im.shape[1])
                ax[k, s].set_ylim(im.shape[0], 0)
                ax[k, s].axis('off')

                # add predicted label
                pred_label = str(self.labels[label_idx[s]]).split(',')[0]
                ax[k, s].text(im.shape[1] / 2, 2.5 if self.line_thickness == 1 else 7.,
                              pred_label, color='black', ha='center', va='center',
                              weight='bold', fontsize=self.fontsize)

                # highlight candidates
                if ((level == 0 and np.any(s == candidates[batch_index]))
                        or (level > 0 and s == candidates[batch_index, k])):
                    highlight_subplot(ax[k, s], plt.get_cmap('coolwarm')(1.), 1 * self.line_thickness)
