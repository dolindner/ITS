import torch
import torch.nn as nn


class NormalizeRotationStrokeBatched(nn.Module):
    """
    PCA-based orientation normalization for 2D stroke sequences.

    Aligns each stroke sequence to its principal axes by rotating the
    absolute coordinates, then converts back to relative deltas.
    Includes options for deterministic eigenvector signing and ensuring
    proper rotation (determinant = +1).
    """

    def __init__(self,
                 max_points: int = -1,
                 sort: bool = False,
                 ensure_proper_rotation: bool = True,
                 fix_sign: bool = False):
        """
        Args:
            max_points (int): The maximum number of points to use for PCA
                from each stroke. If -1, uses all points. Defaults to -1.
            sort (bool): If True, sorts eigenvectors by their corresponding
                eigenvalues in descending order. Defaults to False.
            ensure_proper_rotation (bool): If True, ensures the final
                transformation matrix is a proper rotation (determinant = +1),
                preventing reflections. Defaults to True.
            fix_sign (bool): If True, flips eigenvectors so they align
                consistently with the centroid of each stroke sequence.
                Defaults to False.

        """
        super().__init__()
        self.max_points = max_points
        self.sort = sort
        self.ensure_proper_rotation = ensure_proper_rotation
        self.fix_sign = fix_sign
        self.randomize = True  # for clarity

    @torch.no_grad()
    def forward(self, stroke_seq: torch.Tensor, randomize: bool = None,
                use_svd_for_rotation: bool = False) -> torch.Tensor:
        """
        Args:
            stroke_seq: [B, N, 3] last dim (dx, dy, pen_state)
            randomize: override self.randomize (if None, uses self.randomize)
            use_svd_for_rotation: if True, build orthogonal transform via SVD (U @ Vh)
        Returns:
            [B, N, 3]
        """
        if randomize is None:
            randomize = self.randomize

        B, N, C = stroke_seq.shape
        device = stroke_seq.device
        orig_dtype = stroke_seq.dtype
        compute_dtype = torch.float64  # enforce float64 for all internal math

        # Convert inputs to float64 for computation (but keep pen_state separately)
        coords = stroke_seq[..., :2].to(device=device, dtype=compute_dtype)  # [B,N,2]
        pen_state = stroke_seq[..., 2:].to(device=device, dtype=orig_dtype)  # preserved in original dtype

        # Absolute coordinates
        abs_xy = torch.cumsum(coords, dim=1)  # [B,N,2] in float64

        # Subsample if requested
        if self.max_points <= 0 or N <= self.max_points:
            pos_sel = abs_xy  # [B,N,2]
        else:
            # indices: [B, max_points]
            idx = torch.stack([
                torch.randperm(N, device=device)[:self.max_points]
                for _ in range(B)
            ], dim=0)  # long
            pos_sel = torch.gather(abs_xy, 1, idx.unsqueeze(-1).expand(-1, -1, 2))  # [B, max_points, 2]

        # Mean & centered
        mu = pos_sel.mean(dim=1, keepdim=True)  # [B,1,2]
        centered = pos_sel - mu  # [B, N_sel, 2]

        # Covariance (2x2): (1/n) X^T X
        n_sel = centered.size(1)
        # avoid division by zero, but n_sel should be >=1
        Cmat = torch.bmm(centered.transpose(1, 2), centered) / max(n_sel, 1)  # [B,2,2], float64

        # Regularize covariance to avoid degeneracy (scale eps by trace)
        trace = (torch.diagonal(Cmat, dim1=1, dim2=2).sum(dim=1) / 2.0).clamp(min=1e-12)  # [B]
        eps = (1e-12 + 1e-9 * trace).view(B, 1, 1).to(device=device, dtype=compute_dtype)
        Cmat = Cmat + eps * torch.eye(2, device=device, dtype=compute_dtype).unsqueeze(0)

        # Use symmetric eigen decomposition (more accurate for covariance)
        # eigh returns eigenvalues ascending
        e_vals, e_vecs = torch.linalg.eigh(Cmat)  # e_vecs: [B,2,2], float64

        # Optionally re-order eigenvectors to descending eigenvalue magnitude
        if self.sort:
            idx = e_vals.argsort(dim=-1, descending=True)  # [B,2]
            idx_expand = idx.unsqueeze(1).expand(-1, 2, -1)  # [B,2,2]
            e_vecs = e_vecs.gather(dim=2, index=idx_expand)

        # Optionally use SVD to produce a robust orthogonal matrix
        if use_svd_for_rotation:
            U, S, Vh = torch.linalg.svd(Cmat)  # U: [B,2,2], Vh: [B,2,2]
            R = U @ Vh  # orthogonal, [B,2,2]
            e_vecs = R

        # Randomize (augment) or deterministic sign fixing
        if randomize and self.training:
            # sign flips: generate in float64 then apply
            signs = (torch.randint(0, 2, (B, 2), device=device) * 2 - 1).to(dtype=compute_dtype)  # [-1,1]
            e_vecs = e_vecs * signs.unsqueeze(1)  # broadcast over rows

            if not self.sort:
                swap = torch.rand(B, device=device) > 0.5  # boolean mask
                if swap.any():
                    # swap columns 0 and 1 for those batches
                    swapped = e_vecs.clone()
                    swapped[swap] = e_vecs[swap][:, :, [1, 0]]
                    e_vecs = swapped
        elif self.fix_sign:
            mu_centered = mu.squeeze(1).to(dtype=compute_dtype)  # [B,2]
            dots = torch.einsum("bi,bij->bj", mu_centered, e_vecs)  # [B,2]
            signs = torch.where(dots < 0, -1.0, 1.0).to(dtype=compute_dtype)  # [B,2]
            e_vecs = e_vecs * signs.unsqueeze(1)

        # Ensure proper rotation (determinant +1)
        if self.ensure_proper_rotation:
            det = torch.linalg.det(e_vecs)  # [B]
            flip_mask = det < 0
            if flip_mask.any():
                # flip second column for 2D to correct determinant
                e_vecs[flip_mask, :, 1] *= -1.0

        # Rotate absolute coordinates: abs_xy [B,N,2] * e_vecs [B,2,2] -> [B,N,2]
        rotated_abs = torch.bmm(abs_xy.to(dtype=compute_dtype), e_vecs)  # [B,N,2], float64

        # Convert back to relative deltas
        rel_xy = torch.empty_like(rotated_abs)
        rel_xy[:, 0] = rotated_abs[:, 0]
        if N > 1:
            rel_xy[:, 1:] = rotated_abs[:, 1:] - rotated_abs[:, :-1]

        # Concatenate pen_state back (convert coords back to original dtype)
        rel_xy_out = rel_xy.to(dtype=orig_dtype)
        out = torch.cat([rel_xy_out, pen_state], dim=-1)  # [B,N, C] where C >=3

        return out


class NormalizeToRangeBatched(nn.Module):
    """
    Normalize absolute coordinates to [-128,128] range for stroke sequences.
    """

    def __init__(self, eps: float = 1e-8, scale_abs=False):
        super().__init__()
        self.eps = eps
        self.scale_abs = scale_abs

    def forward(self, stroke_seq: torch.Tensor) -> torch.Tensor:
        # stroke_seq: [B, N, 3] where last dim is (dx, dy, pen_state)
        abs_xy = torch.cumsum(stroke_seq[..., :2], dim=1)  # [B, N, 2]
        center = abs_xy.mean(dim=1, keepdim=True)  # [B, 1, 2]
        centered = abs_xy - center  # [B, N, 2]

        max_abs = centered.abs().amax(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        scale = 128.0 / (max_abs + self.eps)  # [B, 1, 1]
        scaled = centered * scale

        # Convert scaled absolute coordinates back to deltas
        rel_xy = torch.zeros_like(scaled)
        rel_xy[:, 0] = scaled[:, 0]  # First point becomes the delta from origin
        rel_xy[:, 1:] = scaled[:, 1:] - scaled[:, :-1]  # [B, N, 2]

        max_abs_rel = rel_xy.abs().amax(dim=(1, 2), keepdim=True)  # [B, 1, 1]

        pen_state = stroke_seq[..., 2:]  # [B, N, 1] or [B, N, 3]
        return torch.cat([rel_xy, pen_state], dim=-1)  # [B, N, 3] or [B, N, 5]


class ConvertToOneHotPenState(nn.Module):
    """
    Convert single pen state to one-hot encoding (p1, p2, p3).

    Input: (dx, dy, pen_state) where pen_state: 0=pen down, 1=pen up (end of stroke)
    Output: (dx, dy, p1, p2, p3) where:
        - p1=1: pen touching paper (drawing continues) - pen_state=0
        - p2=1: pen lifted after this point (stroke end) - pen_state=1 (last drawn point)
        - p3=1: drawing has ended (not rendered) - point AFTER last pen_state=1 and all subsequent

    The mask is inferred from the pen states: points after the last pen_state=1 get p3=1.
    """

    def __init__(self):
        super().__init__()

    def forward(self, stroke_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stroke_seq: [B, N, 3] with (dx, dy, pen_state)

        Returns:
            [B, N, 5] with (dx, dy, p1, p2, p3)
        """
        dx_dy = stroke_seq[..., :2]  # [B, N, 2]
        pen_state = stroke_seq[..., 2]  # [B, N]

        batch_size, seq_len = stroke_seq.shape[0], stroke_seq.shape[1]

        # Find the last occurrence of pen_state=1 in each sequence
        # This marks the last drawn point
        pen_up_mask = (pen_state == 1).float()  # [B, N]

        # Create position indices
        position_indices = torch.arange(seq_len, device=stroke_seq.device).unsqueeze(0).expand(batch_size, -1)  # [B, N]

        # For each sequence, find the index of the last pen_state=1
        # Set to -1 if no pen_state=1 exists (all drawing)
        pen_up_positions = position_indices * pen_up_mask  # [B, N]
        last_pen_up_idx = pen_up_positions.max(dim=1, keepdim=True).values  # [B, 1]

        # Check if there's at least one pen_state=1 in each sequence
        has_pen_up = pen_up_mask.sum(dim=1, keepdim=True) > 0  # [B, 1]

        # Points AFTER the last pen_state=1 have p3=1
        is_after_last_pen_up = position_indices > last_pen_up_idx  # [B, N]

        # Only set p3=1 if there was actually a pen_state=1 in the sequence
        is_after_last_pen_up = is_after_last_pen_up & has_pen_up  # [B, N]

        # Create one-hot encoding
        # p1: pen down (pen_state=0)
        p1 = (pen_state == 0).float()

        # p2: pen up (pen_state=1) - these are the last points of strokes
        p2 = (pen_state == 1).float()

        # p3: points AFTER the last pen_state=1 (not rendered)
        p3 = is_after_last_pen_up.float()

        return torch.cat([dx_dy, p1.unsqueeze(-1), p2.unsqueeze(-1), p3.unsqueeze(-1)], dim=-1)


class BILSTMSKETCHClassifier(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_classes: int,
                 rnn_type: str = 'lstm',
                 pool_type: str = 'attn',
                 dropout: float = 0.3,
                 num_mlp_layers: int = 1,
                 preprocess_module: nn.Module = None,
                 augmentation: nn.Module = None,
                 attn_dropout: float = 0.2):

        """Bidirectional LSTM/GRU model designed for sketch sequence classification tasks.

            Attributes:
                input_size: Dimension of the input features per sequence step (e.g., 3 or 5).
                hidden_size: Number of features in the hidden state of the RNN.
                num_layers: Number of recurrent layers.
                num_classes: Number of target classification output classes.
                rnn_type: Type of RNN layer to use ('lstm' or 'gru'). Defaults to 'lstm'.
                pool_type: Pooling strategy over time ('attn', 'max', or 'last'). Defaults to 'attn'.
                dropout: Dropout probability between recurrent layers and in the MLP. Defaults to 0.3.
                num_mlp_layers: Number of hidden layers in the final classification MLP. Defaults to 1.
                preprocess_module: Optional nn.Module applied to inputs before the RNN. Defaults to None.
                augmentation: Optional nn.Module for data augmentation. Defaults to None.
                attn_dropout: Dropout probability applied specifically to attention weights. Defaults to 0.2.
            """
        super().__init__()

        assert rnn_type in ['lstm', 'gru']
        assert pool_type in ['attn', 'max', 'last']

        self.hidden_size = hidden_size
        self.pool_type = pool_type
        self.input_size = input_size

        # Preprocessing (optional)
        self.preprocess = preprocess_module or nn.Identity()

        # RNN (bidirectional)
        rnn_class = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            input_size=input_size,  # 3 for (dx,dy,pen_state) or 5 for (dx,dy,p1,p2,p3)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        rnn_output_size = hidden_size * 2

        # Attention pooling
        if self.pool_type == 'attn':
            self.attention = nn.Linear(rnn_output_size, 1, bias=False)
            self.attn_dropout = nn.Dropout(attn_dropout)

        self.augmentation = augmentation
        self.aug = True

        # Classifier (MLP)
        mlp_layers = []
        if num_mlp_layers > 0:
            for i in range(num_mlp_layers):
                in_dim = rnn_output_size if i == 0 else hidden_size * 2
                out_dim = hidden_size * 2
                mlp_layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                    nn.LayerNorm(out_dim),
                    nn.Dropout(dropout),
                ])
            mlp_layers.append(nn.Linear(hidden_size * 2, num_classes))
            self.classifier = nn.Sequential(*mlp_layers)
        else:
            self.classifier = nn.Linear(rnn_output_size, num_classes)

    def _attention_pooling(self, rnn_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # rnn_out: (batch, seq_len, hidden*2)
        attn_scores = self.attention(rnn_out).squeeze(-1)  # (batch, seq_len)

        # Mask padded positions: set them to -inf before softmax
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.nn.functional.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        attn_probs = self.attn_dropout(attn_probs)  # apply dropout on attention distribution

        context = (rnn_out * attn_probs).sum(dim=1)  # weighted sum
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size+1) with format (...features, mask)
        For input_size=3: (dx, dy, pen_state, mask)
        For input_size=5: (dx, dy, p1, p2, p3, mask)
        """
        if self.training and self.augmentation is not None and self.aug:
            x = self.augmentation(x)
        inputs, mask = x[..., :-1], x[..., -1].long()  # (B, T, input_size), (B, T)
        lengths = mask.sum(dim=1).cpu()

        # Preprocess coordinates + pen state(s)
        # ConvertToOneHotPenState no longer needs mask as a separate argument
        inputs = self.preprocess(inputs)

        # inputs now has shape [B, T, input_size] (5 for one-hot, 3 for regular)

        # Pack sequence for RNN
        packed = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        rnn_out, hidden = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=inputs.size(1))

        # Pooling
        if self.pool_type == 'attn':
            pooled = self._attention_pooling(rnn_out, mask)
        elif self.pool_type == 'max':
            rnn_out = rnn_out.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
            pooled, _ = torch.max(rnn_out, dim=1)
        else:  # last
            h_n = hidden[0] if isinstance(self.rnn, nn.LSTM) else hidden
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)

        return self.classifier(pooled)


class StrokeAugment(nn.Module):
    """
    A PyTorch module for applying non-affine augmentations to batched stroke data.

    This class applies a sequence of augmentations suitable for sequence-based
    drawing data (like handwriting or sketches). It first converts the relative
    offsets to absolute coordinates, applies the augmentations, and then converts
    the coordinates back to relative offsets.

    Augmentations:
    1.  **Jitter**: Adds Gaussian noise to each point to simulate hand tremor.
    2.  **Elastic Deformation**: Warps the strokes slightly.
    3.  **Stroke Dropout**: Randomly removes entire strokes (sequences of points
        drawn with the pen down) to train the model on incomplete data.
    """

    def __init__(self,
                 jitter_sigma: float = 1.0,
                 stroke_dropout_prob: float = 0.05,
                 elastic_deformation_alpha: float = 2.0,
                 elastic_deformation_sigma: float = 0.08):
        """
        Initializes the augmentation module.

        Args:
            jitter_sigma (float): Standard deviation of Gaussian noise for jitter.
            stroke_dropout_prob (float): Probability of dropping an entire stroke.
            elastic_deformation_alpha (float): Scaling factor for the magnitude of
                                               elastic deformation displacement.
            elastic_deformation_sigma (float): Standard deviation of the Gaussian
                                               kernel for smoothing the deformation field,
                                               as a fraction of the sequence length.
        """
        super().__init__()
        self.jitter_sigma = jitter_sigma
        self.stroke_dropout_prob = stroke_dropout_prob
        self.elastic_deformation_alpha = elastic_deformation_alpha
        self.elastic_deformation_sigma = elastic_deformation_sigma

    def forward(self, stroke_seq: torch.Tensor) -> torch.Tensor:
        """
        Applies the augmentations to a batch of stroke sequences.

        Args:
            stroke_seq (torch.Tensor): A tensor of shape (B, T, 4) containing
                                       (dx, dy, pen_state, mask).

        Returns:
            torch.Tensor: The augmented stroke sequence tensor.
        """
        if not self.training:
            return stroke_seq
        x, y, pen, mask = stroke_seq.unbind(dim=-1)
        B, T = x.shape
        valid_mask = (mask > 0).float()

        # Convert relative deltas to absolute coordinates
        abs_x = torch.cumsum(x * valid_mask, dim=1)
        abs_y = torch.cumsum(y * valid_mask, dim=1)

        # 1. Apply Jitter
        if self.jitter_sigma > 0:
            noise = torch.randn_like(abs_x) * self.jitter_sigma
            abs_x += noise * valid_mask
            abs_y += noise * valid_mask

        # 2. Apply Elastic Deformation
        if self.elastic_deformation_alpha > 0 and self.elastic_deformation_sigma > 0:
            k_sigma_pixels = T * self.elastic_deformation_sigma
            kernel_size = int(2 * round(k_sigma_pixels * 3)) + 1

            t_range = torch.arange(kernel_size, device=x.device, dtype=torch.float32) - (kernel_size - 1) // 2
            kernel = torch.exp(-t_range ** 2 / (2 * k_sigma_pixels ** 2))
            kernel = (kernel / kernel.sum()).view(1, 1, -1)

            displacement_x = torch.randn(B, 1, T, device=x.device)
            displacement_y = torch.randn(B, 1, T, device=x.device)

            padding = (kernel_size - 1) // 2
            smoothed_dx = torch.nn.functional.conv1d(displacement_x, kernel, padding=padding).squeeze(1)
            smoothed_dy = torch.nn.functional.conv1d(displacement_y, kernel, padding=padding).squeeze(1)

            smoothed_dx = (smoothed_dx / (smoothed_dx.std() + 1e-9)) * self.elastic_deformation_alpha
            smoothed_dy = (smoothed_dy / (smoothed_dy.std() + 1e-9)) * self.elastic_deformation_alpha

            abs_x += smoothed_dx * valid_mask
            abs_y += smoothed_dy * valid_mask

        # Keep track of the final augmented absolute positions
        final_abs_x, final_abs_y = abs_x, abs_y
        final_pen = pen.clone()

        # 3. Apply Stroke Dropout
        if self.stroke_dropout_prob > 0:
            pen_lifted = (pen > 0).float()
            pen_lifted[:, -1] = 1.0

            stroke_ids = torch.cumsum(pen_lifted, dim=1).long() - 1
            stroke_ids.clamp_(min=0)

            num_strokes = stroke_ids[:, -1] + 1
            max_strokes = int(num_strokes.max())

            drop_decisions = torch.rand(B, max_strokes, device=x.device) < self.stroke_dropout_prob

            point_drop_mask = torch.gather(drop_decisions, 1, stroke_ids)
            point_drop_mask[:, 0] = False

            augmented_mask_bool = valid_mask.bool() & ~point_drop_mask

            final_pen[point_drop_mask] = 1.0

            indices = torch.arange(T, device=x.device).repeat(B, 1)
            indices[~augmented_mask_bool] = -1  # Invalidate dropped points' indices

            last_valid_indices = torch.cummax(indices, dim=1).values
            last_valid_indices.clamp_(min=0)  # Ensure indices are valid

            final_abs_x = torch.gather(abs_x, 1, last_valid_indices)
            final_abs_y = torch.gather(abs_y, 1, last_valid_indices)

        # Convert corrected absolute coordinates back to relative deltas
        rel_x = torch.zeros_like(final_abs_x)
        rel_y = torch.zeros_like(final_abs_y)
        rel_x[:, 0] = final_abs_x[:, 0]
        rel_y[:, 0] = final_abs_y[:, 0]
        rel_x[:, 1:] = final_abs_x[:, 1:] - final_abs_x[:, :-1]
        rel_y[:, 1:] = final_abs_y[:, 1:] - final_abs_y[:, :-1]

        return torch.stack([rel_x, rel_y, final_pen, mask], dim=-1)


# Layer mappings for TU Berlin architectures
TU_BERLIN_LAYER_MAPPINGS = {
    "bi_lstm": {
        0: ("classifier.4", "input"),  # last linear layer input
        1: ("classifier.1", "input"),  # GELU input
        2: ("classifier.0", "input"),  # first linear layer input
    },
    "bi_lstm_one_hot": {
        0: ("classifier.4", "input"),  # last linear layer input
        1: ("classifier.1", "input"),  # GELU input
        2: ("classifier.0", "input"),  # first linear layer input
    }
}
