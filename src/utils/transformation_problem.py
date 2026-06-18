import torch

from src.utils.transforms.apply import grid_resample
from src.utils.transform_sequence import create_sampler, create_parameter_sampler, TransformSequence


class TransformationProblem:
    def __init__(self, confidence_module, transform_sequence:TransformSequence, consolidate_method="consolidate_simple",max_batch_size=None):
        """
        A transformation problem is the combination of a transform sequence, that contains
        information about the transformation like methods to apply it and sample it and a confidence module.
        The confidence module can now be used to assign scores to transformed samples. This class provides these helper methods.

        Args:
            confidence_module (nn.Module): Module to calculate confidence for transformed inputs.
            transform_sequence (TransformSequence): TransformSequence object that handles transformations.
            consolidate_method (str, optional): Method to consolidate parallel runs. Defaults to "consolidate_simple".
        """
        self.transform_sequence = transform_sequence
        self.confidence_module = confidence_module
        self.consolidate_method = consolidate_method
        self.max_batch_size = max_batch_size

    def __call__(self, param):
        """
        Calculates the transformation sequence to get a transformation matrix.
        Args:
            param: Parameters for the transformations.
        Returns
            Transformation matrix.
        """
        return self.transform_sequence(param)

    def normalize(self, param):
        """
        Normalize parameters using the underlying transform sequence's normalization rules.
        """
        return self.transform_sequence.normalize(param)

    def transform(self, x, param):
        """Transform the input x using the transformation matrix T.

        Args:
            x (torch.Tensor): Input tensor x.
            param (torch.Tensor): Parameter to transform.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        return self.transform_sequence.transform(x, param)

    def sample_neighbor(self, param, neighboor_hood_size=None):
        """Sample a random point in the neighborhood of the parameter.

        Args:
            param (torch.Tensor): Parameter to sample around.
            neighboor_hood_size (float, optional): Optional scale factor for the neighborhood size.

        Returns:
            torch.Tensor: Random parameter sampled in the neighborhood.
        """
        return self.transform_sequence.sample_neighbor(param, neighboor_hood_size)

    def initial_param(self, batch_size=1, n_samples=None):
        """
        Create an initial parameter for the transformation.

        Args:
            batch_size (int): Batch size.
            n_samples (int): Number of samples per batch.

        Returns:
            torch.Tensor: Initial parameter tensor.
        """
        return self.transform_sequence.initial_param(batch_size, n_samples)

    def correct_param(self, param, reflect=None):
        """
        Corrects the parameter to be within the bounds.

        Args:
            param (torch.Tensor): Parameter to correct.
            reflect (bool, optional): Whether to use reflection for correction.
                If None, uses the transform_sequence default.

        Returns:
            torch.Tensor: Corrected parameter.
        """
        reflect = self.transform_sequence.reflect if reflect is None else reflect
        return self.transform_sequence.correct_param(param, reflect)

    def extract_param_sizes(self):
        """
        Returns the parameter sizes for each transformation in the sequence.
        
        Returns:
            List of parameter sizes
        """
        return self.transform_sequence.extract_param_sizes()
        
    def calc_complete_size(self):
        """
        Calculates the total number of parameters for all transformations.
        
        Returns:
            Total parameter size
        """
        return self.transform_sequence.calc_complete_size()

    def matrix_dim(self) -> int:
        """
        Infers the dimension of the transformation matrix by generating one.
        """
        # Create a zero parameter vector for a single sample
        param_dim = self.get_dim()
        dummy_param = torch.zeros(1, param_dim, device=self.transform_sequence.dummy_param.device, dtype=self.transform_sequence.dummy_param.dtype)
        # Generate a transformation matrix
        T = self.transform_sequence(dummy_param)
        # Return its dimension
        return T.shape[-1]

    def calculate_error(self, x, param, y=None):
        if self.max_batch_size is None or x.size(0) <= self.max_batch_size:
            return self._calculate_error(x, param, y)
        return self._calculate_error_batched(x, param, y)

    def _calculate_error(self, x, param,y=None):
        """
        Calculate the error as the negative confidence. In addition outputs the predicted class.

        Args:
            x (torch.Tensor): Input tensor to transform.
            param (torch.Tensor): Transformation parameter.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the error value and the logits.
        """
        # Apply the transformations
        # Calculate the error using the confidence module
        x = self.transform_sequence.transform(x,param)
        res = self.confidence_module(x, y)
        if isinstance(res, tuple):
            conf, logits = res
            error = -conf
            if logits is None:
                return error, torch.empty(x.size(0), device=x.device).unsqueeze(-1)
            return error, logits
        else:
            return -res,torch.empty(x.size(0), device=x.device).unsqueeze(-1)

    def _calculate_error_batched(self, x, param, y=None):
        """
        Batched error calculation respecting max_batch_size.

        Args:
            x (torch.Tensor): Tensor of shape (B, ...).
            param (torch.Tensor): Tensor of shape (B, ...).
            y (torch.Tensor, optional): Optional tensor of shape (B, ...).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - error: Tensor of shape (B, ...) representing the error.
                - logits: Tensor of shape (B, ...) representing the logits.
        """
        B = x.size(0)
        max_bs = self.max_batch_size or B
        errors = []
        classes = []
        for start in range(0, B, max_bs):
            end = start + max_bs
            xi = x[start:end]
            pi = param[start:end]
            yi = y[start:end] if y is not None else None
            err_chunk, cls_chunk = self._calculate_error(xi, pi, yi)
            errors.append(err_chunk)
            classes.append(cls_chunk)
        error = torch.cat(errors, dim=0)
        if classes[0] is None:
            return error, torch.empty(x.size(0), device=x.device).unsqueeze(-1)
        clss = torch.cat(classes, dim=0)
        return error, clss

    def consolidate_simple(self, x, best_param, best_error, classes_best):
        """
        Consolidate parallel runs by selecting for each sample the run with the minimum error.
        Args:
          best_param: Tensor of shape (batch_size, parallel_runs, param_dim) best paramters
          best_error: Tensor of shape (batch_size, parallel_runs) best error of each run.
          best_other_data: Tensor of shape (batch_size, parallel_runs, class_dim): Likely logits.
        Reeturns:
            Consolidated best_param, best_error, best_other_data (one per sample)
        """
        # Get indices of minimum error per sample
        best_indices = torch.argmin(best_error, dim=1, keepdim=True)  # shape: (batch_size, 1)
        # Gather best parameters: best_param has shape (batch_size, parallel_runs, param_dim)
        best_param_selected = best_param.gather(
            dim=1, 
            index=best_indices.unsqueeze(-1).expand(-1, -1, best_param.size(-1))
        ).squeeze(1)
        # Gather best errors: best_error has shape (batch_size, parallel_runs)
        best_error_selected = best_error.gather(dim=1, index=best_indices).squeeze(1)
        # Gather best classes: classes_best has shape (batch_size, parallel_runs, class_dim)
        best_classes = classes_best.gather(
            dim=1, 
            index=best_indices.unsqueeze(-1).expand(-1, -1, classes_best.size(-1))
        ).squeeze(1)
        return best_param_selected, best_error_selected, best_classes

    def consolidate(self, x, best_param, best_error, classes_best):
        """
        See consolidate simple.
        """
        if self.consolidate_method == "consolidate_simple":
            return self.consolidate_simple(x, best_param, best_error, classes_best)
        else:
            raise ValueError(f"Unknown consolidation method: {self.consolidate_method}")

    def to(self, device):
        """
        Moves all internal tensors to the specified device.
        """
        self.transform_sequence.to(device)
        return self

    def params_to_matrix(self,param):
        """
        Converts the parameter tensor to a transformation matrix.
        Args:
            param: Parameter tensor.
        Retruns:
            Transformation matrix.
        """
        return self.transform_sequence(param)

    def get_dim(self):
        return self.transform_sequence.calc_complete_size()

    def get_identity_parameters(self, batch_size=1):
        """
        Create a parameter that represents the identity transformation.
        Args:
            batch_size: Batch size.
        Returns:
             Identity parameter.
        """
        return self.transform_sequence.get_identity_parameters(batch_size)





if __name__ == "__main__":
    import torch
    from src.utils.transform_sequence import TransformSequence
    from src.utils.affine_transforms import AffineTransformation2D

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a transform sequence
    transforms_2d = [
        AffineTransformation2D.ROTATION.value,
        AffineTransformation2D.TRANSLATION.value
    ]
    domains_2d = [
        (-torch.pi, torch.pi),  # Rotation domain
        ((-1.0, 1.0), (-2.0, 2.0))  # Translation domain
    ]
    
    transform_seq = TransformSequence(transforms_2d, domains_2d, device=device)
    
    # Create a dummy confidence module for testing
    class DummyConfidence:
        def __call__(self, x):
            return torch.sum(x**2, dim=(1,2,3)), torch.ones(x.shape[0], 10).to(x.device)
    
    confidence_module = DummyConfidence()
    
    # Create transformation problem
    problem = TransformationProblem(confidence_module, transform_seq)
    
    # Test initial_param
    param = problem.initial_param(batch_size=2)
    print(f"Initial param shape: {param.shape}")
    
    # Test transformation
    x = torch.randn(2, 3, 28, 28, device=device)
    T = problem(param)
    print(f"Transformation matrix shape: {T.shape}")
    
    # Test transform
    transformed_x = problem.transform(x, param)
    print(f"Transformed x shape: {transformed_x.shape}")
    
    # Test error calculation
    error, classes = problem.calculate_error(x, param)
    print(f"Error: {error}, Classes: {classes}")
    
    # Test new functions
    boundary_violation = problem.boundary_violation(param)
    print(f"Boundary violation: {boundary_violation.shape}")
    
    param2 = problem.initial_param(batch_size=2)
    dist = problem.distance(param, param2)
    print(f"Distance: {dist}")
    
    param_sizes = problem.extract_param_sizes()
    print(f"Parameter sizes: {param_sizes}")
    
    total_size = problem.calc_complete_size()
    print(f"Total parameter size: {total_size}")