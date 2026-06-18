import torch

from search.base_opt import BaseOptimizer


class ParallelGradientDescent(BaseOptimizer):
    def __init__(self, optimizer_class=torch.optim.Adam, optimizer_params=None,
                 learning_rate=0.1, max_iterations=1000, parallel_runs=4,
                 lr_decay_rate=1.0, project_param: bool = True, reflect=True):
        """
        Args:
            learning_rate: Learning rate for gradient descent.
            max_iterations: Maximum number of iterations for gradient descent.
            parallel_runs: Number of parallel runs to perform.
            lr_decay_rate: Rate at which learning rate decays per iteration.
        """
        if optimizer_params is None:
            optimizer_params = {}
        self.optimizer_params = optimizer_params
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.parallel_runs = parallel_runs
        self.lr_decay_rate = lr_decay_rate
        self.project_param = project_param
        self.reflect = reflect

    def optimize(self, transformation_problem, x, y=None, verbose=False):
        """
        Run gradient descent to optimize the transformation parameters.

        Args:
            transformation_problem: An instance of TransformationProblem.
            x: Input image tensor.
            y: Optional targets (will be repeated across parallel runs if provided).
            verbose: Whether to print progress information.

        Returns:
            Tuple (best_param, best_error, best_other_data)
        """
        batch_size = x.shape[0]
        total_batches = self.parallel_runs * batch_size
        current_param_pre = transformation_problem.initial_param(batch_size, self.parallel_runs)
        current_param = current_param_pre.reshape(total_batches, -1).requires_grad_(True)
        x_repeated = x.repeat_interleave(self.parallel_runs, dim=0)
        y_repeated = y.repeat_interleave(self.parallel_runs, dim=0) if y is not None else None

        # extract max_batch_size for chunked computation to save vram
        max_chunk = transformation_problem.max_batch_size if transformation_problem.max_batch_size is not None else total_batches

        optimizer = self.optimizer_class([current_param], lr=self.learning_rate, **self.optimizer_params)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=self.lr_decay_rate) if self.lr_decay_rate != 1.0 else torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1.0)

        # set best values to none, other is used to keep track of class of best
        best_param, best_error, best_other = None, None, None
        with torch.enable_grad():
            for iteration in range(self.max_iterations):
                optimizer.zero_grad(set_to_none=True)

                p_old = current_param.detach().clone()

                # accumulate gradients per chunk to avoid keeping the full graph in memory
                total = current_param.shape[0]
                all_errors = []
                all_others = []

                for start in range(0, total, max_chunk):
                    end = min(start + max_chunk, total)

                    # compute error for the chunk; do NOT collect all chunks before backward
                    err_chunk, cls_chunk = transformation_problem._calculate_error(
                        x_repeated[start:end],
                        current_param[start:end],
                        y=y_repeated[start:end] if y_repeated is not None else None
                    )

                    err_chunk.mean().backward()

                    all_errors.append(err_chunk.detach())
                    all_others.append(cls_chunk.detach())

                # concatenate current errors before optimizer step
                error = torch.cat(all_errors, dim=0)
                other = torch.cat(all_others, dim=0)
                e_old = error.clone()

                del all_errors, all_others

                # check that grad is not None
                if current_param.grad is None:
                    raise ValueError(
                        "Gradient is None. Check the transformation problem and ensure gradients are being computed correctly.")
                optimizer.step()

                with torch.no_grad():
                    # Project or normalize params
                    if self.project_param:
                        current_param.data = transformation_problem.correct_param(current_param,
                                                                                  reflect=self.reflect).data
                    else:
                        current_param.data = transformation_problem.normalize(current_param).data

                    if iteration == 0:
                        best_param = p_old.clone()
                        best_error = error.detach().clone()
                        best_other = other.detach().clone()
                    else:
                        improved = error.detach() < best_error
                        best_param[improved] = p_old[improved]
                        best_error[improved] = error.detach()[improved]
                        best_other[improved] = other.detach()[improved]

                    if verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                        lr = scheduler.get_last_lr()[0]
                        print(
                            f"Iter {iteration + 1}/{self.max_iterations}, LR:{lr:.6f}, Err:{e_old.mean():.4f}, Best:{best_error.mean():.4f}")
                    scheduler.step()

        # final evaluation on last params
        with torch.no_grad():
            final_error, final_other = transformation_problem.calculate_error(x_repeated, current_param, y=y_repeated)
            improved_final = final_error < best_error
            best_param[improved_final] = current_param[improved_final]
            best_error[improved_final] = final_error[improved_final]
            best_other[improved_final] = final_other[improved_final]

        # reshape
        return self._reshape_results(x, best_param, best_error, best_other, transformation_problem)

    def _reshape_results(self, x, params, error, other, problem):
        params = params.view(x.shape[0], self.parallel_runs, -1)
        error = error.view(x.shape[0], self.parallel_runs)
        other = other.view(x.shape[0], self.parallel_runs, -1)
        return problem.consolidate(x, params, error, other)
