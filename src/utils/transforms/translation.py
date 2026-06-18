import torch

from src.utils.helper import identity
from src.utils.transforms.bounded_transform import BoundedTransform


class Translation(BoundedTransform):
    """Arbitrary D‐dimensional translation."""

    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        # Check if the input is a tensor
        if not isinstance(param, torch.Tensor):
            raise ValueError("Input parameter must be a torch.Tensor")

        # Get the dimension of the parameter

        dim = param.shape[-1]
        batch_size = param.shape[:-1]
        # Create a translation matrix
        translation_matrix = identity(batch_size, dim + 1, dtype=param.dtype, device=param.device)
        translation_matrix[..., :dim, -1] = param
        return translation_matrix

    def param_size(self) -> int:
        return self.dims


class DirectedTranslation(BoundedTransform):
    """Translation along one axis in D dimensions."""

    def __init__(self, dims: int, axis: int):
        super().__init__()
        self.dims = dims
        self.axis = axis

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        # build full vector then inline translation
        vec = torch.zeros(*param.shape[:-1], self.dims, dtype=param.dtype, device=param.device)
        vec[..., self.axis] = param.squeeze(-1)
        batch = vec.shape[:-1]
        I = torch.eye(self.dims + 1, dtype=vec.dtype, device=vec.device)
        T = I.expand(*batch, self.dims + 1, self.dims + 1).clone()
        T[..., :self.dims, -1] = vec
        return T

    def param_size(self) -> int:
        return 1


# instantiate common transforms_old
Translate2D = Translation(2)
Translate3D = Translation(3)
TranslateX2D = DirectedTranslation(2, 0)
TranslateY2D = DirectedTranslation(2, 1)
TranslateX3D = DirectedTranslation(3, 0)
TranslateY3D = DirectedTranslation(3, 1)
TranslateZ3D = DirectedTranslation(3, 2)

if __name__ == "__main__":
    import torch
    from src.utils.transforms.apply import grid_resample, transform_3d_point_cloud

    # Test Translation 2D with images
    param_img = torch.randn(1, 2, requires_grad=True)
    matrix_img = Translate2D.matrix(param_img)
    x_img = torch.randn(1, 1, 28, 28)
    res_img = grid_resample(x_img, matrix_img)
    res_img.sum().backward()
    assert param_img.grad is not None, "Image gradient is None"
    assert param_img.grad.abs().sum().item() > 0, "Image gradient is zero"

    # Numeric gradient check for 2D
    param_img_d = torch.randn(1, 2, dtype=torch.double, requires_grad=True)
    x_img_d = x_img.to(torch.double)
    fn_img = lambda p: grid_resample(x_img_d, Translate2D.matrix(p))
    assert torch.autograd.gradcheck(fn_img, (param_img_d,), eps=1e-6, atol=1e-4), "Image gradcheck failed"

    # Test Translation 3D with point cloud
    param_pc = torch.randn(1, 3, requires_grad=True)
    matrix_pc = Translate3D.matrix(param_pc)
    x_pc = torch.randn(1, 1024, 3)
    out_pc = transform_3d_point_cloud(x_pc, matrix_pc)
    out_pc.sum().backward()
    assert param_pc.grad is not None, "Point cloud gradient is None"
    assert param_pc.grad.abs().sum().item() > 0, "Point cloud gradient is zero"

    # Numeric gradient check for 3D
    param_pc_d = torch.randn(1, 3, dtype=torch.double, requires_grad=True)
    x_pc_d = x_pc.to(torch.double)
    fn_pc = lambda p: transform_3d_point_cloud(x_pc_d, Translate3D.matrix(p))
    assert torch.autograd.gradcheck(fn_pc, (param_pc_d,), eps=1e-6, atol=1e-4), "Point cloud gradcheck failed"

    # Test Directed Translation 2D (X-axis)
    param_dir_x = torch.randn(1, 1, requires_grad=True)
    matrix_dir_x = TranslateX2D.matrix(param_dir_x)
    res_dir_x = grid_resample(x_img, matrix_dir_x)
    res_dir_x.sum().backward()
    assert param_dir_x.grad is not None, "Directed X-axis gradient is None"
    assert param_dir_x.grad.abs().sum().item() > 0, "Directed X-axis gradient is zero"

    # Test Directed Translation 3D (Z-axis)
    param_dir_z = torch.randn(1, 1, requires_grad=True)
    matrix_dir_z = TranslateZ3D.matrix(param_dir_z)
    out_dir_z = transform_3d_point_cloud(x_pc, matrix_dir_z)
    out_dir_z.sum().backward()
    assert param_dir_z.grad is not None, "Directed Z-axis gradient is None"
    assert param_dir_z.grad.abs().sum().item() > 0, "Directed Z-axis gradient is zero"

    print("All translation tests passed!")
