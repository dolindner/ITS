import torch

from search.base_opt import BaseOptimizer
from src.utils.transformation_problem import TransformationProblem


class PSO(BaseOptimizer):
    """
    Particle Swarm Optimization (Kennedy & Eberhart, 1995).
    Population (swarm) evolves via inertia, cognitive, social terms.

    Args:
        swarm_size: Number of particles in the swarm.
        steps: Number of optimization steps (iterations).
        w: Inertia weight controlling momentum.
        c1: Cognitive weight for personal best influence.
        c2: Social weight for global best influence.
        clamp_velocity: Whether to clamp velocity to prevent divergence.
        v_max_scale: Max velocity as a fraction of initial parameter span.
        project_param: Whether to project parameters back into valid domain after updates.
    """

    def __init__(
            self,
            swarm_size: int = 64,
            steps: int = 30,
            w: float = 0.72,
            c1: float = 1.49,
            c2: float = 1.49,
            clamp_velocity: bool = True,
            v_max_scale: float = 0.2,
            project_param: bool = True,
    ):
        self.swarm_size = swarm_size
        self.steps = steps
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.clamp_velocity = clamp_velocity
        self.v_max_scale = v_max_scale
        self.project_param = project_param

    def optimize(self, transformation_problem: TransformationProblem, x, y=None, verbose=False):
        with torch.no_grad():
            device = x.device
            batch = x.shape[0]

            # initialize swarm positions from the same sampler used in SHGO
            pos = transformation_problem.initial_param(batch, self.swarm_size)  # (B,S,D)
            B, S, D = pos.shape
            vel = torch.zeros_like(pos)

            # evaluate initial
            flat_pos = pos.view(-1, D)
            x_rep = x.repeat_interleave(S, dim=0)
            y_rep = y.repeat_interleave(S, dim=0) if y is not None else None
            pbest_err, pbest_other = transformation_problem.calculate_error(x_rep, flat_pos, y=y_rep)
            pbest_err = pbest_err.view(B, S)
            pbest_other = pbest_other.view(B, S, -1)
            pbest_pos = pos.clone()

            # global best per batch
            g_idx = torch.argmin(pbest_err, dim=1)  # (B,)
            gbest_pos = pbest_pos[torch.arange(B, device=device), g_idx]  # (B,D)
            gbest_err = pbest_err[torch.arange(B, device=device), g_idx]  # (B,)
            gbest_other = pbest_other[torch.arange(B, device=device), g_idx]  # (B, D_other)

            # precompute velocity clamp bounds using initial spanned values as a heuristic.
            if self.clamp_velocity:
                span = (pos.max(dim=1).values - pos.min(dim=1).values).clamp(min=1e-6)  # (B,D)
                v_max = self.v_max_scale * span  # (B,D)

            for it in range(self.steps):
                r1 = torch.rand_like(pos)
                r2 = torch.rand_like(pos)

                cognitive = self.c1 * r1 * (pbest_pos - pos)
                social = self.c2 * r2 * (gbest_pos.unsqueeze(1) - pos)
                vel = self.w * vel + cognitive + social

                if self.clamp_velocity:
                    # broadcast v_max (B,1,D)
                    vmax = v_max.unsqueeze(1)
                    vel = torch.clamp(vel, -vmax, vmax)

                pos = pos + vel
                # project / correct or normalize. Project means that for domain restricted transformation we reproject into the domain.
                # normalize does not reproject into domain but only normalizes transformation that require it(like normalized quaternions)
                flat_pos = pos.view(-1, D)
                if self.project_param:
                    flat_pos = transformation_problem.correct_param(flat_pos)
                else:
                    flat_pos = transformation_problem.normalize(flat_pos)
                pos = flat_pos.view(B, S, D)

                # evaluate current error
                flat_pos = pos.view(-1, D)
                curr_err, curr_other = transformation_problem.calculate_error(x_rep, flat_pos, y=y_rep)
                curr_err_v = curr_err.view(B, S)
                curr_other_v = curr_other.view(B, S, -1)

                improved = curr_err_v < pbest_err
                if improved.any():
                    pbest_err = torch.where(improved, curr_err_v, pbest_err)
                    pbest_pos = torch.where(improved.unsqueeze(-1), pos, pbest_pos)
                    pbest_other = torch.where(improved.unsqueeze(-1), curr_other_v, pbest_other)

                # update global best
                g_idx = torch.argmin(pbest_err, dim=1)
                gbest_pos = pbest_pos[torch.arange(B, device=device), g_idx]
                gbest_err = pbest_err[torch.arange(B, device=device), g_idx]
                gbest_other = pbest_other[torch.arange(B, device=device), g_idx]

            # final values are the tracked global bests
            final_params = gbest_pos.unsqueeze(1)  # (B,1,D)
            final_err = gbest_err.view(B, 1)
            final_other = gbest_other.view(B, 1, -1)

            return transformation_problem.consolidate(x, final_params, final_err, final_other)
