import torch
class BaseOptimizer:
    def optimize(
        self,
        transformation_problem,
        x: torch.Tensor,
        y: torch.Tensor = None,
        verbose: bool = False
    ):
        """
        Given a transformation problem(transformations plus confidence model) it optmizes an input
        x over the space defined by the transformations. Optionally labels can be passed which will
        be passed to confidence model. Mostly for debug purposes.
        If verbose the optimizer may print progress and other debug information.
        """