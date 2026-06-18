from abc import ABC, abstractmethod
from typing import Optional

import torch


class Transform(ABC):
    """
    Subclasses must implement:
      - matrix(self, param: Tensor) -> Tensor  # the (batch, D+1, D+1) matrix
      - param_size(self) -> int  # number of parameters
      - if support_calc_bounds() is True:
            calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> (Tensor, Tensor)  # lower, upper bounds for each parameter
    """

    def __init__(self):
        """
        Initializes the Transform class.
        :param log: If True, orbits will be in log-space, otherwise in linear space.
        """
        pass

    @abstractmethod
    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Function that creates a transformation matrix treating the last dimension as the translation vector.
        Args:
            param: The parameter tensor. The output will use param's dtype and device.

        Returns:
            torch.Tensor: The transformation matrix with shape (..., D+1, D+1).
        """
        ...

    @abstractmethod
    def param_size(self) -> int:
        """
        Returns the size of the parameter vector.
        """
        ...

    @abstractmethod
    def sample_param(self, batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """
        Sample a parameter for the transformation.
        """
        ...

    def project_parameters(self, param: torch.Tensor, domain, reflect=True) -> torch.Tensor:
        """
        Args:
            param: param to reflect
            reflect:  If True, reflect the parameters into the domain. If False, clip the parameters to the domain.
        Returns:  The projected parameters.
        """
        return param

    def normalize_parameters(self, param: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the parameters. Used for quaternions to keep at unit norm.
        Args:
            param: param to noramlize.
        Returns:  The normalized parameters.
        """
        return param

    @abstractmethod
    def supports_sobol(self) -> bool:
        """
        Indicates if the transform supports sobol sampling.
        """
        ...

    def sample_space_param_size(self):
        """
        Tells special sample methods like sobol how many parameters they should sample.
        Only required if the transform supports sobol sampling.
        """
        return self.param_size()

    # methods that have to be potentially overridden by subclasses
    def sobol_to_param(self, sparam: torch.Tensor, domain) -> torch.Tensor:
        """
        Converts parameters from the range [0,1] to actual parameters.
        """
        pass

    @abstractmethod
    def supports_orbit(self) -> bool:
        """
        Indicates if the transform supports orbit sampling.
        True if the transform supports orbit sampling, False otherwise.
        """

    def orbit(self,
              n_samples: int,
              domain,
              dim=0,
              extend: int = 0,
              shift: int = 0) -> Optional[torch.Tensor]:
        """
        Generates a set of samples along the orbit of a discrete transformation. Only works for Transforms that use a single parameter.
        Other transforms_old return None. Per default it reuses sobol to paramter and generates samples from
        0 to 1 in linear space and then converts them to parameters.
        Args:
            n_samples: Number of smaples along the orbit.
            domain: Domain of the transformation
            dim: Number dimesions
            extend:Sample additional members over the boundary.
            shift: Shift the linspace untested and unused.

        Returns:

        """

        # get per-dimension bounds and determine parameter dimension

        low_p = 0
        high_p = 1

        # total samples including padding
        total = n_samples + 2 * extend

        # linear spacing in [low_p, high_p]
        rng = high_p - low_p
        spacing = rng / (n_samples - 1) if n_samples > 1 else 0
        start = low_p - extend * spacing
        values = torch.linspace(start,
                                high_p + extend * spacing,
                                total) + shift * spacing
        # project values back into [0,1] via modulo (finish previous TODO)
        if total > 0:
            values = torch.remainder(values, 1.0)
        sample_dim = self.sample_space_param_size()
        params = torch.zeros((total, sample_dim), dtype=torch.float32)
        params[:, dim] = values

        # convert params from sample space to parameter space
        params = self.sobol_to_param(params)
        return params

    # methods that should not be overridden by subclasses
    def as_dict(self):
        """
        Provide a dictionary interface for backward compatibility.
        """
        return {
            "matrix": self.apply,
            "param_size": self.param_size(),
            "project_parameters": self.project_parameters,
            "calc_bounds": self.calc_bounds,
            "orbit": [self.orbit, ],
            "param": [self.orbit_member, ],
            "interval": self.interval(),
        }

    def __call__(self, param: torch.Tensor) -> torch.Tensor:
        """
        Calls the matrix method with the given parameter.
        Args:
            param: The parameter to create the transformation matrix for.
        Returns:
            The transformation matrix.
        """
        return self.matrix(param)

    def interval(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns the interval of the parameters if it is fixed, otherwise None.
        By default, returns None, indicating no fixed interval.
        Subclasses can override this to provide specific intervals for their parameters.
        """
        return None

    def __getitem__(self, key: str):
        """
        Allow backward-compatible dict-style access, e.g. transform['matrix'].
        """
        return self.as_dict()[key]

    def apply(self, T: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformation described by param to a previous transformation matrix using matrix multiplication.
        Args:
            T: Transformation matrix.
            param: if T is non this is used to create T.
        Returns:
         The transformed data.
        """
        if T is None:
            return self.matrix(param)
        # matrix multiplication
        return torch.matmul(self.matrix(param), T)

    def __eq__(self, other):
        # Only compare if they are exactly the same class
        if self.__class__ is not other.__class__:
            return NotImplemented
        # Compare each attribute in __dict__; both keys and values must match
        return self.__dict__ == other.__dict__

    def __hash__(self):
        items = tuple(sorted(self.__dict__.items()))
        return hash((self.__class__, items))

    def orbit_member(self, n: int, n_samples: int, domain, dim=0) -> torch.Tensor:
        """ gets a specific orbit member."""
        res = self.orbit(n_samples=n_samples, domain=domain, dim=dim, extend=0, shift=0)
        return res[n]

    # TODO consider removing
    # consider removing and only implementing this for bounded transforms_old
    @abstractmethod
    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the bounds given a domain parameter. This is a fallback for search methods that do not support projecting into the bounds.
        Not all transforms_old support this.
        Args:
            domain: The domain to calculate bounds for
            dtype: Optional dtype for the output tensors (defaults to torch.float32 if None)
            device: Optional device for the output tensors (defaults to 'cpu' if None)

        Returns:
            The bounds of the transformation.
        """
        ...

    def support_calc_bounds(self) -> bool:
        """
        Indicates if the transform implements calc_bounds_fallback.
        """
        return False

    def default_neighbourhood_size(self, domain=None, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Return a default neighborhood size for this transform in parameter space.
        By default returns 1 for each parameter.
        Args:
            domain: optional domain info to pick sensible scale
            dtype, device: tensor creation options
        Returns:
            1D tensor of length self.param_size() with nonnegative neighborhood sizes.
        """
        # default conservative small neighborhood
        size = self.param_size()
        return torch.ones(size, dtype=dtype, device=device)

    def num_discrete_values(self) -> Optional[int]:
        """
        Returns the number of discrete values this transformation can take.
        If infinite (continuous), returns None.
        Subclasses should override if discrete.
        """
        return None

    def identity_param(self, batch_size: Optional[int] = 1, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Returns the parameter that corresponds to the identity transformation.
        By default, returns a zero tensor.
        Returns:
            A tensor of shape (param_size,) corresponding to the identity transformation.
        """
        size = self.param_size()
        return torch.zeros((batch_size, size), dtype=dtype, device=device)
