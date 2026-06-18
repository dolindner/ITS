import torch


def _build_param_index_map(param_sizes):
    """
    Given a list of param_sizes per transform, return a list of (transform_idx, inner_idx)
    for each flat parameter index.
    """
    mapping = []
    for t_idx, size in enumerate(param_sizes):
        for inner in range(size):
            mapping.append((t_idx, inner))
    return mapping


class CoordinateDescent:
    def __init__(self, number_samples: int = 10):
        """
        Initializes a simple coordinate descent optimization problem. Uses internal sampling of transformation problem
        to sample each dimension.

        Args:
            number_samples: int, number of random samples per dimension

        """
        self.number_samples = number_samples

    def optimize(
            self,
            transformation_problem,
            x: torch.Tensor,
            y: torch.Tensor = None,
            verbose: bool = False
    ):
        device = x.device
        batch_size = x.size(0)
        n_samples = self.number_samples

        # 1) draw multiple random samples per dim: (batch, samples, dim)
        random_params = transformation_problem.initial_param(batch_size, n_samples)
        random_params = random_params.view(batch_size, n_samples, -1).to(device)
        dim = random_params.size(2)

        # prepare discreteness mapping (per-flat-dimension). This is used for transforms that have only a discrete number of elements like reflection. Here we dont require random sampling.
        ts = transformation_problem.transform_sequence
        discreteness = ts.get_discreteness_vector().to(torch.long).cpu().tolist()
        param_sizes = ts.extract_param_sizes()
        param_to_transform = _build_param_index_map(param_sizes)

        # 2) initialize best parameters and their error
        best_params = transformation_problem.get_identity_parameters(batch_size).to(device)
        best_err = torch.full((batch_size,), float('inf'), device=device)
        best_other = None

        # 3) per-dimension search
        for d in range(dim):
            # default candidate values from random sampling
            cand_vals = random_params[:, :, d]  # (batch, n_samples)
            cand_count = n_samples

            # If coordinate is discrete and has few discrete values, use orbit sampling
            n_disc = discreteness[d]
            t_idx, inner_idx = param_to_transform[d]
            transform = ts.transformations[t_idx]
            domain = ts.domains[t_idx] if t_idx < len(ts.domains) else None

            # Use transform.supports_orbit() as authoritative flag (no hasattr checks)
            supports_orbit = transform.supports_orbit()

            if n_disc is not None and n_disc > 0 and n_disc <= n_samples and supports_orbit:
                # request orbit samples (n_disc)
                vals = transform.orbit(n_samples=n_disc, domain=domain, dim=inner_idx)
                if vals is not None:
                    vals = torch.as_tensor(vals, device=device, dtype=random_params.dtype)
                    if vals.dim() == 1:
                        vals = vals.unsqueeze(-1)
                    if vals.size(-1) > 1:
                        vals_coord = vals[:, inner_idx].reshape(-1)
                    else:
                        vals_coord = vals.reshape(-1)
                    cand_vals = vals_coord.unsqueeze(0).expand(batch_size, -1).to(device)
                    cand_count = cand_vals.size(1)

            trials = best_params.unsqueeze(1).expand(-1, cand_count, -1).clone()
            trials[:, :, d] = cand_vals

            flat_trials = trials.reshape(-1, dim)
            x_rep = x.repeat_interleave(cand_count, dim=0)
            y_rep = y.repeat_interleave(cand_count, dim=0) if y is not None else None

            with torch.no_grad():
                errs, others = transformation_problem.calculate_error(x_rep, flat_trials, y_rep)
                errs = errs.view(batch_size, cand_count)
                others = others.view(batch_size, cand_count, -1)

            if best_other is None:
                best_other = torch.zeros(batch_size, others.size(-1), device=device, dtype=others.dtype)

            best_vals, best_idx = errs.min(dim=1)
            mask = best_vals < best_err
            if mask.any():
                # pick values from cand_vals using best_idx, being careful about broadcasting
                chosen = cand_vals.contiguous()[mask, best_idx[mask]]
                best_params[mask, d] = chosen
                best_err[mask] = best_vals[mask]
                best_other[mask] = others[mask, best_idx[mask]]

        # final evaluation and consolidation
        final_params = best_params.unsqueeze(1)
        final_err = best_err.view(batch_size, 1)
        final_other = best_other.view(batch_size, 1, -1)

        return transformation_problem.consolidate(
            x, final_params, final_err, final_other
        )


class WeightedCoordinateDescent:
    def __init__(self, samples_per_dim: list[int], rounds: int = 1):
        """

        Initializes a weighted coordinate descent optimization problem. Uses internal sampling of transformation problem
        to sample each dimension. Here one can vary the samples per dimension. The reason is that one
        can typically choose dimensions that the network is the least robust against. For images rotation is one such candidate.
        During hyperparam opt we typically only vary the number of samples in the first dimesion vs the rest due to the number
        of hyperparameters being dependant on the number of dims otherwise.

        Args:
            samples_per_dim: list of ints, number of random samples to draw per dimension
            rounds: number of full passes over all dimensions
        """
        self.samples_per_dim = samples_per_dim
        self.rounds = rounds

    def optimize(
            self,
            transformation_problem,
            x: torch.Tensor,
            y: torch.Tensor = None,
            verbose: bool = False
    ):
        device = x.device
        batch_size = x.size(0)

        # determine dimension
        _, _, dim = transformation_problem.initial_param(batch_size, 1) \
            .view(batch_size, 1, -1).shape
        assert len(self.samples_per_dim) == dim, \
            f"Expected {dim} samples_per_dim, got {len(self.samples_per_dim)}"

        # prepare discreteness mapping (do not hide errors)
        ts = transformation_problem.transform_sequence
        discreteness = ts.get_discreteness_vector().to(torch.long).cpu().tolist()
        param_sizes = ts.extract_param_sizes()
        param_to_transform = _build_param_index_map(param_sizes)

        # initialize best params, error, and other
        best_params = transformation_problem.get_identity_parameters(batch_size).to(device)
        best_err = torch.full((batch_size,), float('inf'), device=device)
        best_other = None

        # multiple rounds of per-dimension search
        for _ in range(self.rounds):
            # draw per-dimension candidates each round (respect discreteness/orbit)
            cand_vals_per_dim = []
            cand_counts = []
            for d, n in enumerate(self.samples_per_dim):
                n_disc = discreteness[d]
                t_idx, inner_idx = param_to_transform[d]
                transform = ts.transformations[t_idx]
                domain = ts.domains[t_idx] if t_idx < len(ts.domains) else None

                supports_orbit = transform.supports_orbit()

                if n_disc is not None and n_disc > 0 and n_disc <= n and supports_orbit:
                    vals = transform.orbit(n_samples=n_disc, domain=domain, dim=inner_idx)
                    if vals is not None:
                        vals = torch.as_tensor(vals, device=device,
                                               dtype=transformation_problem.initial_param(1, 1).dtype)
                        if vals.dim() == 1:
                            vals = vals.unsqueeze(-1)
                        if vals.size(-1) > 1:
                            vals_coord = vals[:, inner_idx].reshape(-1)
                        else:
                            vals_coord = vals.reshape(-1)
                        cand = vals_coord.unsqueeze(0).expand(batch_size, -1).to(device)
                        cand_vals_per_dim.append(cand)
                        cand_counts.append(cand.size(1))
                        continue  # next dimension

                # fallback: sample using initial_param
                full = transformation_problem.initial_param(batch_size, n)
                full = full.view(batch_size, n, -1).to(device)
                cand = full[:, :, d]
                cand_vals_per_dim.append(cand)
                cand_counts.append(cand.size(1))

            # per-dimension update
            for d in range(dim):
                cand = cand_vals_per_dim[d]
                ccount = cand.size(1)
                trials = best_params.unsqueeze(1) \
                    .expand(-1, ccount, -1) \
                    .clone()
                trials[:, :, d] = cand

                flat = trials.reshape(-1, dim)
                x_rep = x.repeat_interleave(ccount, dim=0)
                y_rep = y.repeat_interleave(ccount, dim=0) if y is not None else None

                with torch.no_grad():
                    errs, others = transformation_problem.calculate_error(
                        x_rep, flat, y_rep
                    )
                    errs = errs.view(batch_size, ccount)
                    others = others.view(batch_size, ccount, -1)

                if best_other is None:
                    best_other = torch.zeros(batch_size, others.size(-1), device=device, dtype=others.dtype)

                best_vals, best_idx = errs.min(dim=1)
                mask = best_vals < best_err
                if mask.any():
                    best_params[mask, d] = cand[mask, best_idx[mask]]
                    best_err[mask] = best_vals[mask]
                    best_other[mask] = others[mask, best_idx[mask]]

        # consolidate final results
        final_params = best_params.unsqueeze(1)
        best_err = best_err.view(batch_size, 1)
        best_other = best_other.view(batch_size, 1, -1)
        return transformation_problem.consolidate(
            x, final_params, best_err, best_other
        )
