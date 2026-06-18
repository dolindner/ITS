import torch
import math
from src.utils.transformation_problem import TransformationProblem

from search.base_opt import BaseOptimizer


class ParallelSimulatedAnnealing(BaseOptimizer):
    def __init__(
        self,
        initial_temp=100.0,
        cooling_rate=0.95,
        max_iterations=1000,
        parallel_runs=4,
        reinit_interval=9999999999999,
        reinit_amount=0.0,
        project_param=True,
        neighbor_hood_size=0.1,
    ):
        """
        Args:
            initial_temp: Initial temperature for simulated annealing.
            cooling_rate: Cooling rate for simulated annealing.
            max_iterations: Maximum number of iterations.
            parallel_runs: Number of parallel runs.
            reinit_interval: How often to reinitialize worst runs.
            reinit_amount: Fraction (0-1) or int number of worst runs to reinit.
            project_param: Whether to project parameters after neighbor sampling.
            neighbor_hood_size: Step size for neighbor sampling.
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.parallel_runs = parallel_runs
        self.reinit_interval = reinit_interval
        self.reinit_amount = reinit_amount
        self.project_param = project_param
        self.neighbor_hood_size = neighbor_hood_size

    def _detect_and_reinit_worst(self, iteration, current_param, current_error, transformation_problem, verbose):
        """
        Detect and reinitialize worst-performing runs.
        Returns updated (current_param, current_error, reinit_mask).
        """
        with torch.no_grad():
            total = current_param.shape[0]
            reinit_mask = torch.zeros(total, dtype=torch.bool, device=current_param.device)

            if (iteration + 1) % self.reinit_interval == 0:
                # determine number of runs to reinitialize
                if isinstance(self.reinit_amount, float):
                    amount = math.floor(self.parallel_runs * self.reinit_amount)
                else:
                    amount = self.reinit_amount

                if amount > 0:
                    batch_size = total // self.parallel_runs
                    current_param_reshaped = current_param.view(batch_size, self.parallel_runs, -1)
                    current_error_reshaped = current_error.view(batch_size, self.parallel_runs)

                    worst_indices = current_error_reshaped.argsort(dim=1)[:, -amount:]

                    new_params = transformation_problem.initial_param(batch_size, amount).to(current_param.device)
                    new_error = torch.full((batch_size, amount), float("inf"), device=current_error.device)

                    batch_indices = torch.arange(batch_size, device=current_param.device).unsqueeze(1)
                    current_param_reshaped[batch_indices, worst_indices] = new_params
                    current_error_reshaped[batch_indices, worst_indices] = new_error

                    # set mask
                    for b in range(batch_size):
                        reinit_mask[b * self.parallel_runs + worst_indices[b]] = True

                    if verbose:
                        print(f"Iteration {iteration + 1}: Reinitialized {amount} worst runs per sample.")

                    current_param = current_param_reshaped.view(total, -1)
                    current_error = current_error_reshaped.view(total)

            return current_param, current_error, reinit_mask

    def optimize(self, transformation_problem: TransformationProblem, x, y=None, verbose=False):
        """
        Run parallel simulated annealing to optimize transformation parameters.
        """
        with torch.no_grad():
            batch_size = x.shape[0]
            total_batches = self.parallel_runs * batch_size

            # Initialize parameters
            current_param = transformation_problem.initial_param(batch_size, self.parallel_runs)
            current_param = current_param.reshape(total_batches, -1)

            x_repeated = x.repeat_interleave(self.parallel_runs, dim=0)
            y_repeated = y.repeat_interleave(self.parallel_runs, dim=0) if y is not None else None

            current_error, other_data = transformation_problem.calculate_error(x_repeated, current_param,y=y_repeated)
            best_param = current_param.clone()
            best_error = current_error.clone()
            best_other_data = other_data.clone()

            temp = self.initial_temp

            for iteration in range(self.max_iterations):
                # Detect and reinit worst runs, get mask
                current_param, current_error, reinit_mask = self._detect_and_reinit_worst(
                    iteration, current_param, current_error, transformation_problem, verbose
                )

                # Sample neighbors only for non-reinit runs
                active_mask = ~reinit_mask
                neighbor_param = current_param.clone()
                if active_mask.any():
                    neighbor_param[active_mask] = transformation_problem.sample_neighbor(
                        current_param[active_mask], neighboor_hood_size=self.neighbor_hood_size
                    )

                if self.project_param:
                    neighbor_param = transformation_problem.correct_param(neighbor_param)
                else:
                    neighbor_param = transformation_problem.normalize(neighbor_param)

                neighbor_error, neighbor_other = transformation_problem.calculate_error(
                    x_repeated, neighbor_param, y=y_repeated
                )

                # Acceptance
                accept = (neighbor_error < current_error) | \
                         (torch.exp((current_error - neighbor_error) / temp) > torch.rand_like(current_error))

                current_param = torch.where(accept.unsqueeze(-1), neighbor_param, current_param)
                current_error = torch.where(accept, neighbor_error, current_error)

                update_best = neighbor_error < best_error
                best_param = torch.where(update_best.unsqueeze(-1), neighbor_param, best_param)
                best_error = torch.where(update_best, neighbor_error, best_error)
                best_other_data = torch.where(update_best.unsqueeze(-1), neighbor_other, best_other_data)

                temp *= self.cooling_rate

                if verbose:
                    print(f"Iteration {iteration + 1}/{self.max_iterations}, "
                          f"Temp: {temp:.4f}, Mean Best Error: {best_error.mean():.4f}, "
                          f"Current Error: {current_error.mean():.4f}")

            # Reshape outputs to (batch_size, parallel_runs, ...)
            best_param = best_param.view(batch_size, self.parallel_runs, -1)
            best_error = best_error.view(batch_size, self.parallel_runs)
            best_other_data = best_other_data.view(batch_size, self.parallel_runs, -1)

            # Consolidate best run per sample
            best_param, best_error, best_other_data = transformation_problem.consolidate(
                x, best_param, best_error, best_other_data
            )

            return best_param, best_error, best_other_data