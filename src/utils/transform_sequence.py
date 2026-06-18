from copy import deepcopy

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from src.utils.transforms.apply import grid_resample

from src.utils.transforms.periodic_transform import PeriodicTransform
def create_parameter_sampler(transform_sequence):
    """
    Create a parameter sampler for the given transform sequence.
    Args
        transform_sequence: A TransformSequence object.
    Returns
         A function that takes a batch size and returns sampled parameters.
    """
    def sampler(batch_size):
        return transform_sequence.sample_individual(batch_size)
    return sampler


def create_sampler(transform_sequence):
    """
    Create a transform matrix sampler for the given transform sequence.
     Args
        transform_sequence: A TransformSequence object.
    Returns:
         A function that takes a batch size and returns transformation matrices.
    """

    def sampler(batch_size):
        params = transform_sequence.sample_individual(batch_size)
        return transform_sequence(params)
    return sampler

class TransformSequence (torch.nn.Module):
    def __init__(self, transformations, domains, neighbour_hood_size=None,
                 application_method=grid_resample, device="cpu", dtype=torch.float32,
                 init_method="individual", use_individual_param_correction=False,
                 reflect=False, invert=False):
        """
        Creates a sequence of individual transforms. Provides various helper methods
        to call them.
        Args:
            transformations (List[Callable]): List of transformation functions. See affine_transforms.py for details.
            domains (List[Tuple]): List of tuples representing the domain for each transformation.
            neighbour_hood_size (List[float]): List of floats indicating the size of the neighborhood for each transformation.
            application_method (Callable, optional): Function to apply the transformation. Defaults to grid_resample.
            device (str, optional): Device to use for computations. Defaults to "cpu".
            init_method (str, optional): Method to initialize parameters. Options are "sobol", "latin_hypercube", "uniform", "individual".
                "individual" uses the sample method of the class which samples each parameter of each transformation separately.
                This may avoid sampling outside the domain depending on how the transformation interprets its domain (for example, hypersphere for skew rotation).
                The disadvantage is that space-filling sampling is not possible. So reprojecting a space-filling method may still be preferred if reflect is True. Defaults to "sobol".
            reflect (bool, optional): If True, reflect the parameters on the boundaries; otherwise, clamp them. Defaults to False.
        """
        # list of transformations to apply in order.
        super().__init__()
        self.transformations = transformations
        self.domains = domains
        self.dummy_param = torch.nn.Parameter(torch.zeros(1,device=device,dtype=dtype), requires_grad=False,)

        self.application_method = application_method


        self.sizes = self.extract_sampling_param_sizes()


        self.reflect = reflect
        self.invert = invert  # whether to invert the transformation



        self.init_method = init_method


        if neighbour_hood_size is None:
            # Build per-transform default neighbourhoods and concatenate
            nh_list = []
            for i, transform in enumerate(self.transformations):
                dom = self.domains[i] if i < len(self.domains) else None
                nh_list.append(transform.default_neighbourhood_size(domain=dom,
                                                                    dtype=self.dummy_param.dtype,
                                                                    device=self.dummy_param.device))
            nh = torch.cat(nh_list, dim=-1) if len(nh_list) > 0 else torch.ones(self.calc_complete_size(), dtype=dtype, device=device)
        else:
            nh = TransformSequence.calc_neighbor_hood_size(neighbour_hood_size, self.sizes, device, dtype)
        self.register_buffer("neighbour_hood_size", nh)


        self.to(device)



    def extract_sampling_param_sizes(self):
        """
        Extracts the original parameter sizes from the transformations.
        Returns: List of parameter sizes for each transformation.
        """
        return [transformation.sample_space_param_size() for transformation in self.transformations]


    @staticmethod
    def calc_neighbor_hood_size(neighborhood_size, param_sizes, device="cpu", dtype=torch.float32):
        """Calculates the neighborhood size tensor for each parameter value so that one can sample neighbors.

        Args:
            neighborhood_size (Union[float, List[float]]): Single value or list of values for neighborhood size.
            param_sizes (List[int]): List of parameter sizes for each transformation.
            device (torch.device): Device on which to create the tensor.

        Returns:
            torch.Tensor: Tensor of shape (total_param_size,) with neighborhood size values.
        """
        # if it is a tensor interpret as a per parameter
        if isinstance(neighborhood_size, torch.Tensor) or isinstance(neighborhood_size, np.ndarray):
            # check if it is a tensor of the correct size
            if isinstance(neighborhood_size, np.ndarray):
                neighborhood_size = torch.tensor(neighborhood_size, dtype=dtype, device=device)
            if neighborhood_size.shape[0] != sum(param_sizes):
                raise ValueError(f"Neighborhood size tensor must have the same size as the total parameter size. "
                                 f"Expected {sum(param_sizes)}, got {neighborhood_size.shape[0]}.")
            return neighborhood_size

        # interpret as iterable of individual neighborhood sizes per transformation that has to extended to match inidival number param
        nh_list = []


        if not isinstance(neighborhood_size, (list, tuple)):
            neighborhood_size = [neighborhood_size] * len(param_sizes)


        for i, size in enumerate(param_sizes):
            nh_val = neighborhood_size[i]
            # if type is tensor directly use it otherwise create full tensor
            if isinstance(nh_val, torch.Tensor):
                nh_list.append(nh_val)
            else:
                nh_list.append(torch.tensor([nh_val] * size, dtype=dtype, device=device))

        # Concatenate all tensors to create a single tensor with one value per parameter
        return torch.cat(nh_list, dim=-1)


    def extract_param_sizes(self):
        """
        Calculates the number of parameters for all transformations.
        Returns: List with the number of paramters for each transformation
        """
        return [transformation["param_size"] for transformation in self.transformations]

    def calc_complete_size(self):
        """
        Calculates the total number of parameters for all transformations.
        Returns: Total number of parameters.
        """
        return sum(self.extract_param_sizes())

    def correct_param(self, param, reflect=False):
        """
        Corrects the parameter to be within the bounds using clamping or reflection.
        Periodic parameters are wrapped into the default interval before applying the correction.
        This should also normalize the parameters, responsibility is for the transforms to implement this.

        Args:
            param (torch.Tensor or list): The parameter tensor to correct. If a list, it is concatenated.
            reflect (bool): If True, reflect the parameters on the boundaries; otherwise, clamp them.

        Returns:
            torch.Tensor: The corrected parameter tensor.
        """

        if isinstance(param, torch.Tensor):
            # split using sizes
            sizes = self.extract_param_sizes()
            param = torch.split(param, sizes, dim=-1)
            param = list(param)

        for i, transformation in enumerate(self.transformations):
            # get the transformation function
            cr = transformation.project_parameters
            # apply the transformation function to the parameter

            param[i] = cr(param[i], self.domains[i], reflect=reflect)
        par = torch.cat(param, dim=-1)
        return par

    def normalize(self, param):
        """
        Apply each transform's normalize_parameters to its parameter block.
        Keeps tensors split/concat behavior similar to correct_param.
        """

        if isinstance(param, torch.Tensor):
            sizes = self.extract_param_sizes()
            parts = list(torch.split(param, sizes, dim=-1))
        else:
            parts = list(param)

        for i, transformation in enumerate(self.transformations):
            parts[i] = transformation.normalize_parameters(parts[i])

        return torch.cat(parts, dim=-1)

    def __call__(self, param, layer: int | None = None):
        """Applies the transformations in order to calculate the final transformation matrix.

            Args:
                param (Union[torch.Tensor, List[torch.Tensor]]): Parameter tensor to split/process, or a pre-split list of parameters.
                layer is deprecated remove.
            Returns:
                torch.Tensor: The final calculated (and potentially inverted) transformation matrix T.
        """
        # if type is list expect it to be split already. Otherwise we split the tensor.
        if isinstance(param, torch.Tensor):
            sizes = self.extract_param_sizes()
            param = torch.split(param, sizes, dim=-1)

        # now apply the transformations in order to calculate the final transformation matrix
        # create the transformation matrix
        T = None
        for i, transformation in enumerate(self.transformations):
            # get the transformation function
            transform_func = transformation["matrix"]
            T = transform_func(T, param[i])
        if self.invert:
            T = torch.linalg.inv(T)
        return T

    def transform(self, x, param):
        """Transform the input x using the transformation matrix T.

        Args:
            x (torch.Tensor): Input image of shape (3, H, W), (1, H, W), or (4, H, W).
            param (torch.Tensor): Parameter tensor used to transform the input.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        T = self(param)
        return self.application_method(x, T)

    def get_dim(self):
        """
        Get the total number of parameters for all transformations.
        Returns: Total number of parameters.
        """
        return self.calc_complete_size()

    def sample_neighbor(self, param, neighbour_hood_size=None):
        """
        Sample a random point in the neighborhood of the parameter.
        Args:
            param: Parameter to sample around.
        Returns:
            Random parameter in the neighborhood.
        """

        if neighbour_hood_size is None:
            neighbour_hood_size = self.neighbour_hood_size
        else:
            neighbour_hood_size = self.neighbour_hood_size * neighbour_hood_size

        # check if param is a tensor
        if isinstance(param, torch.Tensor):
            noise = torch.rand_like(param) * neighbour_hood_size - neighbour_hood_size / 2
            param = param + noise
        else:
            param_tensor = torch.cat(param, dim=-1)
            noise = torch.rand_like(param_tensor) * neighbour_hood_size - neighbour_hood_size / 2
            param = param_tensor + noise

        # Apply transform-specific normalization (e.g., push zeros to ±eps for reflections)
        param = self.normalize(param)
        return param

    def sample_individual(self,batch_size,n_samples=None, reflect=False):
        """
        Sample a random point in the parameter space using each transformation's sampling method.

        Args:
            use_fallback_correction (bool): Whether to use fallback correction method instead of
                                           transformation-specific projection.
            reflect (bool): If using fallback correction, whether to reflect or clamp parameters.

        Returns:
            torch.Tensor: Sampled parameter vector.
        """
        param = []
        total_param_size = batch_size * n_samples if n_samples is not None else batch_size
        for i, transformation in enumerate(self.transformations):
            # get the transformation's sampling method
            p = transformation.sample_param(total_param_size,self.domains[i],device=self.dummy_param.device,dtype=self.dummy_param.dtype)
            p2 = transformation.project_parameters(p, self.domains[i],reflect=reflect)
            #test to se that difference is not bigger than 1e-4
            assert torch.allclose(p, p2, atol=1e-4,rtol=1e-4), f"Sampled parameter not correctly projected for transformation {i}."
            param.append(p2)

        param = torch.cat(param, dim=-1)

        if n_samples is not None:
            # reshape to (batch_size, n_samples, param_size)
            param = param.view(batch_size, n_samples, -1)
        return param

    def sobol_to_param(self, sparam: torch.Tensor) -> torch.Tensor:
        """
        Convert concatenated sample-space parameters in [0,1] to actual parameter space
        using each transformation's sobol_to_param.
        Supports shapes:
          (B, D_sample) or (B, N, D_sample)
        """
        sample_sizes = self.extract_sampling_param_sizes()
        # Flatten leading dims except last
        original_shape = sparam.shape
        last_dim = original_shape[-1]
        if last_dim != sum(sample_sizes):
            raise ValueError(f"Provided sample dimension {last_dim} != expected {sum(sample_sizes)}")
        flat = sparam.view(-1, last_dim)
        split = torch.split(flat, sample_sizes, dim=-1)
        converted = [transform.sobol_to_param(block,domain)
                     for block, transform, domain in zip(split, self.transformations, self.domains)]
        # Concatenate back

        out = torch.cat(converted, dim=-1)
        # Reshape back
        out = out.view(*original_shape[:-1], out.shape[-1])
        return out


    def get_sampling_dim(self):
        return sum(self.extract_sampling_param_sizes())

    @staticmethod
    def sample_initial_parameters(batch_size, dim, n_samples=None, device=None, dtype=None):
        """
        Sample initial parameters uniformly in [0, 1].
        Args:
            batch_size: Number of batches to sample
            dim: Number of parameters per sample
            n_samples: Number of samples per batch (optional)
            device: Device to create tensors on
            dtype: Data type for tensors

        Returns:
            A tensor of shape (batch, n_samples, dim) with sampled parameters if n_samples is provided, otherwise (batch, dim).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        if n_samples is None:
            return torch.rand(batch_size, dim, device=device, dtype=dtype)
        return torch.rand(batch_size, n_samples, dim, device=device, dtype=dtype)

    @staticmethod
    def sample_initial_parameters_latin_hypercube(batch_size, dim, n_samples=None, device=None, dtype=None):
        """
        Sample parameters using Latin Hypercube Sampling in [0, 1].
        """
        if n_samples is None:
            return TransformSequence.sample_initial_parameters(batch_size, dim, device=device, dtype=dtype)
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        segments = torch.stack([
            torch.stack([torch.randperm(n_samples, device=device).float()
                         for _ in range(dim)], dim=1)
            for _ in range(batch_size)
        ], dim=0)
        u = torch.rand(batch_size, n_samples, dim, device=device, dtype=dtype)
        return (segments + u) / n_samples

    @staticmethod
    def sample_initial_parameters_sobol(batch_size, dim, n_samples=None, device=None, dtype=None):
        """
        Sample parameters using Sobol sequence in [0, 1].
        """
        if n_samples is None:
            return TransformSequence.sample_initial_parameters(batch_size, dim, device=device, dtype=dtype)
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        engine = SobolEngine(dimension=dim, scramble=True)
        sob = engine.draw(batch_size * n_samples).to(device=device, dtype=dtype)
        return sob.view(batch_size, n_samples, dim)

    @staticmethod
    def sample_initial_parameters_permuted_lattice(batch_size, dim, n_samples=None, device=None, dtype=None):
        """Optimized version using argsort for fast permutations"""
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        offsets = torch.rand(batch_size, dim, 1, device=device, dtype=dtype) / n_samples
        base_seq = (offsets + torch.arange(n_samples, device=device, dtype=dtype).view(1, 1, -1) / n_samples) % 1


        # Generate random values and use argsort for permutations (MUCH faster!)
        perm = torch.rand(batch_size, dim, n_samples, device=device, dtype=dtype).argsort(dim=2)
        samples = torch.gather(base_seq, 2, perm)

        samples = samples.permute(0, 2, 1)

        return samples

    def initial_param(self, batch_size=1, n_samples=None):
        """
        Initialize parameters using the requested strategy.
        For sobol/latin_hypercube/uniform: sample in [0,1] then convert via sobol_to_param.
        For 'individual': call per-transform sampling (may already output actual param space).
        """
        sample_dim = self.get_sampling_dim()
        device = self.dummy_param.device
        dtype = self.dummy_param.dtype

        if self.init_method == "sobol":
            s = self.sample_initial_parameters_sobol(batch_size, sample_dim, n_samples, device=device, dtype=dtype)
            return self.sobol_to_param(s)
        elif self.init_method == "latin_hypercube":
            s = self.sample_initial_parameters_latin_hypercube(batch_size, sample_dim, n_samples, device=device, dtype=dtype)
            return self.sobol_to_param(s)
        elif self.init_method == "uniform":
            s = self.sample_initial_parameters(batch_size, sample_dim, n_samples, device=device, dtype=dtype)
            return self.sobol_to_param(s)
        elif self.init_method == "permuted_lattice":
            s = self.sample_initial_parameters_permuted_lattice(batch_size, sample_dim, n_samples, device=device, dtype=dtype)
            #make contiguous
            s = s.contiguous()
            return self.sobol_to_param(s)
        elif self.init_method == "individual":
            return self.sample_individual(batch_size, n_samples, reflect=self.reflect)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

    def copy(self, device=None):
        """
        Return a (deep) copy of this TransformSequence, optionally moved to a device.
        """
        new_obj = deepcopy(self)
        if device is not None:
            new_obj.to(device)
        return new_obj


    def get_identity_parameters(self, batch_size=1,):
        """
        Get the parameters that correspond to the identity transformation.
        :param batch_size: Number of batches to create.
        :return: Parameters corresponding to the identity transformation.
        """
        param = []
        for i, transformation in enumerate(self.transformations):
            # get the transformation's identity parameter
            p = transformation.identity_param(batch_size, device=self.dummy_param.device, dtype=self.dummy_param.dtype)
            param.append(p)
        param = torch.cat(param, dim=-1)
        return param



    def get_inverted_sequence(self):
        """Create a transformation sequence returning the inverted transformation matrix.

        Args:
            transforms (List[Callable]): List of transformations to invert.
            domains (List[Tuple]): List of domains for each transformation.
            device (torch.device): Device to use for computations.
            dtype (torch.dtype): Data type for tensors.

        Returns:
            TransformSequence: Inverted TransformSequence object.
        """
        #create a deepcopy
        inverted = deepcopy(self)
        inverted.invert = not inverted.invert  # toggle invert flag
        return inverted


    def get_discreteness_vector(self):
        """
        Returns a vector of length equal to the total parameter size.
        For each parameter:
          - If the corresponding transform is continuous, value is -1.
          - If the transform is discrete (has num_discrete_values), value is the number of discrete samples.
        """
        discreteness = []
        for i, t in enumerate(self.transformations):
            n = -1
            val = t.num_discrete_values()
            if val is not None:
                n = int(val)
            size = t.param_size()
            discreteness.extend([n] * size)
        return torch.tensor(discreteness, dtype=torch.long, device=self.dummy_param.device)

if __name__ == "__main__":
    import torch
    from src.utils.affine_transforms import AffineTransformation2D, AffineTransformations3D

    # Set up device and dtype for testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    print(f"Testing on device: {device} with dtype: {dtype}")

    # Test 1: Create a 2D transform sequence with rotation and translation
    print("\n=== Test 1: 2D Rotation + Translation ===")
    transforms_2d = [
        AffineTransformation2D.ROTATION.value,
        AffineTransformation2D.TRANSLATION.value
    ]
    domains_2d = [
        (-torch.pi, torch.pi),  # Rotation domain
        ((-1.0, 1.0), (-2.0, 2.0))  # Translation domain
    ]


    seq_2d = TransformSequence(transforms_2d, domains_2d, device=device, dtype=dtype)
    par_angle = seq_2d.initial_param(batch_size=1000)
    # Test with normal and fallback correction methods
    print("Testing with normal correction method...")
    params_normal = seq_2d.sample_individual(1000)
    print(f"Parameters: {params_normal}")

    seq_2d2 = seq_2d.get_version_optimized_for_prediction()

    # Already done: initial_param with batch size 1000
    params = seq_2d2.initial_param(1000)
    print(f"Initial parameters shape: {params.shape}")

    # Sample individual parameters
    individual_params = seq_2d2.sample_individual(1000)
    print(f"Individual sampled params shape: {individual_params.shape}")

    # Correct parameters (with and without reflection)
    corrected_params = seq_2d2.correct_param(params, reflect=True)
    corrected_params_clamp = seq_2d2.correct_param(params, reflect=False)
    print(f"Corrected params (reflect) shape: {corrected_params.shape}")
    print(f"Corrected params (clamp) shape: {corrected_params_clamp.shape}")

    # Get transformation matrices
    transform_matrices = seq_2d2(params[:5])
    print(f"Transformation matrices shape: {transform_matrices.shape}")

    # Apply transformation to dummy image data
    dummy_image = torch.randn(1, 3, 64, 64, device=device)
    transformed = seq_2d2.transform(dummy_image, params[0:1])
    print(f"Transformed image shape: {transformed.shape}")


    # Check boundary violations
    violations = seq_2d2.boundary_violation(params[:10])
    print(f"Boundary violations shape: {violations.shape}, mean: {violations.mean()}")

    # Test parameter distance calculation
    dist = seq_2d2.distance(params[0:5], params[5:10])
    print(f"Distance between parameters: {dist}")

    # Get inverted sequence
    inverted_seq = seq_2d2.get_inverted_sequence()
    print(f"Created inverted sequence, invert flag: {inverted_seq.invert}")

    # Check parameter sizes
    param_sizes = seq_2d2.extract_param_sizes()
    sampling_sizes = seq_2d2.extract_sampling_param_sizes()
    total_size = seq_2d2.calc_complete_size()
    print(f"Parameter sizes: {param_sizes}")
    print(f"Sampling parameter sizes: {sampling_sizes}")
    print(f"Complete parameter size: {total_size}")

    #benchmark sample sobol
    #set sobol
    seq_2d2.init_method = "sobol"
    import time
    start_time = time.time()
    for _ in range(4000):
        _ = seq_2d2.initial_param(batch_size=1000, n_samples=1)
    end_time = time.time()
    print(f"Benchmark: 1000 iterations of initial_param with batch_size=1000, n_samples=1 took {end_time - start_time:.4f} seconds.")
