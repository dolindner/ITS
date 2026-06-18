import torch

from src.utils.helper import identity
from src.utils.transforms.apply import grid_resample, transform_3d_point_cloud
from src.utils.transforms.bounded_transform import BoundedTransform


class Shear2D(BoundedTransform):
    """General 2D shear with two parameters."""

    def __init__(self):
        super().__init__()
        self.dims = 2

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Computes a homogeneous 2D shear matrix.
        Expects 'param' with last dimension of length 2: [shear_x, shear_y].
        """
        batch_size = param.shape[:-1]
        H = identity(batch_size, 3, dtype=param.dtype, device=param.device)
        sh_x = param[..., 0]
        sh_y = param[..., 1]
        H[..., 0, 1] = sh_x
        H[..., 1, 0] = sh_y
        return H

    def param_size(self) -> int:
        return 2

    def orbit(self, n_samples: int, domain, dim=0, extend: int = 0, shift: int = 0):
        # Multi-param transform: no single-parameter orbit
        return None


class Shear3D(BoundedTransform):
    """General 3D shear with six parameters."""

    def __init__(self):
        super().__init__()
        self.dims = 3

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Computes a homogeneous 3D shear matrix.
        Expects 'param' with last dimension of length 6 corresponding to:
          [sh_xy, sh_yx, sh_xz, sh_zx, sh_yz, sh_zy]
        """
        batch_size = param.shape[:-1]
        sh_xy = param[..., 0]
        sh_yx = param[..., 1]
        sh_xz = param[..., 2]
        sh_zx = param[..., 3]
        sh_yz = param[..., 4]
        sh_zy = param[..., 5]
        H = identity(batch_size, 4, dtype=param.dtype, device=param.device)
        H[..., 0, 1] = sh_xy
        H[..., 1, 0] = sh_yx
        H[..., 0, 2] = sh_xz
        H[..., 2, 0] = sh_zx
        H[..., 1, 2] = sh_yz
        H[..., 2, 1] = sh_zy
        return H

    def param_size(self) -> int:
        return 6

    def orbit(self, n_samples: int, domain, extend: int = 0, shift: int = 0, dim=0):
        # Multi-param transform: no single-parameter orbit
        return None


class ShearSequential(BoundedTransform):
    """Computes a shear matrix via sequential composition of local shear matrices."""

    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        num_params = param.shape[-1]
        # Calculate dimension directly: n*(n-1)=num_params -> n = (1+sqrt(1+4*num_params))/2
        n = int((1 + (1 + 4 * num_params) ** 0.5) / 2)

        batch_size = param.shape[:-1]
        K = num_params  # Total number of local shear matrices

        # Create a tensor of identity matrices for each local shear matrix:
        I = torch.eye(n, dtype=param.dtype, device=param.device)
        local_matrices = I.expand(batch_size + (K, n, n)).clone()

        # Get indices for off-diagonals in row-major order
        eye_mask = torch.eye(n, dtype=torch.bool, device=param.device)
        all_idx = torch.nonzero(~eye_mask, as_tuple=False)  # shape (n*(n-1), 2)
        order = (all_idx[:, 0] * n + all_idx[:, 1]).argsort()
        all_idx = all_idx[order]
        i_idx, j_idx = all_idx[:, 0], all_idx[:, 1]

        # Assign each shear parameter to its corresponding off-diagonal position
        # local_matrices has shape (..., K, n, n), param has shape (..., K)
        local_matrices[..., torch.arange(K), i_idx, j_idx] = param

        # Sequentially multiply all local shear matrices
        matrices_list = torch.unbind(local_matrices, dim=-3)  # unbind the K dimension
        S = matrices_list[0]
        for i in range(1, K):
            S = torch.matmul(S, matrices_list[i])

        H = torch.eye(n + 1, dtype=param.dtype, device=param.device).expand(batch_size + (n + 1, n + 1)).clone()
        H[..., :n, :n] = S
        return H

    def param_size(self) -> int:
        return self.dims * (self.dims - 1)

    def orbit(self, n_samples: int, domain, extend: int = 0, shift: int = 0, dim=0):
        # Multi-param transform: no single-parameter orbit
        return None


class ShearFull(BoundedTransform):
    """Computes a shear matrix by directly filling in all off-diagonal entries.
    Not really a shear.
    """

    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        num_params = param.shape[-1]
        n = None
        for potential in range(2, 100):
            if potential * (potential - 1) == num_params:
                n = potential
                break
        if n is None:
            raise ValueError("Invalid number of shear parameters for full shear.")

        batch_size = param.shape[:-1]
        S = identity(batch_size, n, dtype=param.dtype, device=param.device)
        idx = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    S[..., i, j] = param[..., idx]
                    idx += 1
        H = identity(batch_size, n + 1, dtype=param.dtype, device=param.device)
        H[..., :n, :n] = S
        return H

    def param_size(self) -> int:
        return self.dims * (self.dims - 1)

    def orbit(self, n_samples: int, domain, extend: int = 0, shift: int = 0, dim=0):
        # Multi-param transform: no single-parameter orbit
        return None


class DirectedShear(BoundedTransform):
    """Shear directed by a specific parameter in a specific dimension."""

    def __init__(self, dims: int, parameter_index: int):
        super().__init__()
        self.dims = dims
        self.parameter_index = parameter_index

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        degrees = int(self.dims * (self.dims - 1))
        shear_params = torch.zeros(*param.shape[:-1], degrees, dtype=param.dtype, device=param.device)
        shear_params[..., self.parameter_index] = param[..., 0]

        if self.dims == 2:
            return Shear2D().matrix(shear_params)
        elif self.dims == 3:
            return Shear3D().matrix(shear_params)
        else:
            return ShearFull(self.dims).matrix(shear_params)

    def param_size(self) -> int:
        return 1

    def orbit(self, n_samples: int, domain, extend: int = 0, shift: int = 0, dim=0) -> torch.Tensor:
        # Reuse calc_bounds to properly parse domain
        low_p, high_p = self.calc_bounds(domain, dtype=torch.float32, device="cpu")
        low_p, high_p = low_p.squeeze(), high_p.squeeze()

        # Create orbit samples
        total_samples = n_samples + 2 * extend
        spacing = (high_p - low_p) / (n_samples - 1) if n_samples > 1 else 0
        start = low_p - extend * spacing
        end = high_p + extend * spacing
        orbit = torch.linspace(start, end, total_samples, dtype=torch.float32, device="cpu")
        if total_samples == 1:
            orbit = torch.tensor([(start + end) / 2], dtype=torch.float32, device="cpu")

        orbit = orbit + shift * spacing
        res = torch.zeros(total_samples, 1, dtype=orbit.dtype, device=orbit.device)
        res[..., 0] = orbit
        return res


class ShearScaled(BoundedTransform):
    """Shear repesented by L @ D @ L^T  (LTDL form). First paramtes are entries of D else are entries of L"""

    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Returns SPD matrix for first `dims` dimensions,
        then augments to (dims+1)x(dims+1) for affine transforms_old.
        """
        batch_shape = param.shape[:-1]
        dim_affine = self.dims + 1  # full affine size

        # --- L @ D @ L^T for first `dims` ---
        diag = param[..., :self.dims]
        D = torch.diag_embed(torch.exp(diag))  # positive diagonal

        L = torch.eye(self.dims, device=param.device, dtype=param.dtype).expand(*batch_shape, self.dims,
                                                                                self.dims).clone()
        tril_indices = torch.tril_indices(row=self.dims, col=self.dims, offset=-1)
        L[..., tril_indices[0], tril_indices[1]] = param[..., self.dims:]

        SPD_core = L @ D @ L.transpose(-1, -2)

        # --- augment to (dims+1)x(dims+1) for affine ---
        SPD = torch.eye(dim_affine, device=param.device, dtype=param.dtype).expand(*batch_shape, dim_affine,
                                                                                   dim_affine).clone()
        SPD[..., :self.dims, :self.dims] = SPD_core

        return SPD

    def param_size(self) -> int:
        # Only params for the first `dims` dimensions
        return self.dims + (self.dims * (self.dims - 1)) // 2

    def orbit(self, n_samples: int, domain, extend: int = 0, shift: int = 0, dim=0):
        return None


# Instantiate common transforms_old
Shear2DGeneral = Shear2D()
Shear3DGeneral = Shear3D()
ShearSequential2D = ShearSequential(2)
ShearSequential3D = ShearSequential(3)
ShearFull2D = ShearFull(2)
ShearFull3D = ShearFull(3)
ShearX2D = DirectedShear(2, 0)
ShearY2D = DirectedShear(2, 1)
ShearXY3D = DirectedShear(3, 0)
ShearYX3D = DirectedShear(3, 1)
ShearXZ3D = DirectedShear(3, 2)
ShearZX3D = DirectedShear(3, 3)
ShearYZ3D = DirectedShear(3, 4)
ShearZY3D = DirectedShear(3, 5)

ShearScaled2D = ShearScaled(2)
ShearScaled3D = ShearScaled(3)

if __name__ == "__main__":
    print("Testing gradient flow through class-based shear functions...")

    # ------------ Test 1: 2D Shear Matrix ------------
    print("\n1. Testing 2D shear matrix with images:")
    param_2d = torch.randn(1, 2, requires_grad=True)
    matrix_2d = Shear2DGeneral.matrix(param_2d)
    x_img = torch.randn(1, 1, 28, 28)
    res_2d = grid_resample(x_img, matrix_2d)
    res_2d.sum().backward()
    assert param_2d.grad is not None, "2D shear gradient is None"
    assert param_2d.grad.abs().sum().item() > 0, "2D shear gradient is zero"
    print("✓ Manual gradient check passed")

    # Numeric gradient check
    param_2d_d = torch.randn(1, 2, dtype=torch.double, requires_grad=True)
    x_img_d = x_img.to(torch.double)
    fn_2d = lambda p: grid_resample(x_img_d, Shear2DGeneral.matrix(p))
    assert torch.autograd.gradcheck(fn_2d, (param_2d_d,), eps=1e-6, atol=1e-4), "2D shear gradcheck failed"
    print("✓ Numeric gradient check passed")

    # ------------ Test 2: 3D Shear Matrix ------------
    print("\n2. Testing 3D shear matrix with point cloud:")
    param_3d = torch.randn(1, 6, requires_grad=True)  # 6 parameters for 3D shear
    matrix_3d = Shear3DGeneral.matrix(param_3d)
    x_pc = torch.randn(1, 1024, 3)
    out_3d = transform_3d_point_cloud(x_pc, matrix_3d)
    out_3d.sum().backward()
    assert param_3d.grad is not None, "3D shear gradient is None"
    assert param_3d.grad.abs().sum().item() > 0, "3D shear gradient is zero"
    print("✓ Manual gradient check passed")

    # Numeric gradient check
    param_3d_d = torch.randn(1, 6, dtype=torch.double, requires_grad=True)
    x_pc_d = x_pc.to(torch.double)
    fn_3d = lambda p: transform_3d_point_cloud(x_pc_d, Shear3DGeneral.matrix(p))
    assert torch.autograd.gradcheck(fn_3d, (param_3d_d,), eps=1e-6, atol=1e-4), "3D shear gradcheck failed"
    print("✓ Numeric gradient check passed")

    # ------------ Test 3: Sequential Shear Matrix with 2D ------------
    print("\n3. Testing sequential shear matrix with 2D images:")
    param_seq_2d = torch.randn(1, 2, requires_grad=True)  # 2 parameters for 2D sequential shear
    matrix_seq_2d = ShearSequential2D.matrix(param_seq_2d)
    res_seq_2d = grid_resample(x_img, matrix_seq_2d)
    res_seq_2d.sum().backward()
    assert param_seq_2d.grad is not None, "Sequential 2D shear gradient is None"
    assert param_seq_2d.grad.abs().sum().item() > 0, "Sequential 2D shear gradient is zero"
    print("✓ Manual gradient check passed")

    # Numeric gradient check
    param_seq_2d_d = torch.randn(1, 2, dtype=torch.double, requires_grad=True)
    fn_seq_2d = lambda p: grid_resample(x_img_d, ShearSequential2D.matrix(p))
    assert torch.autograd.gradcheck(fn_seq_2d, (param_seq_2d_d,), eps=1e-6,
                                    atol=1e-4), "Sequential 2D shear gradcheck failed"
    print("✓ Numeric gradient check passed")

    # ------------ Test 4: Sequential Shear Matrix with 3D ------------
    print("\n4. Testing sequential shear matrix with 3D point cloud:")
    param_seq_3d = torch.randn(1, 6, requires_grad=True)  # 6 parameters for 3D sequential shear
    matrix_seq_3d = ShearSequential3D.matrix(param_seq_3d)
    out_seq_3d = transform_3d_point_cloud(x_pc, matrix_seq_3d)
    out_seq_3d.sum().backward()
    assert param_seq_3d.grad is not None, "Sequential 3D shear gradient is None"
    assert param_seq_3d.grad.abs().sum().item() > 0, "Sequential 3D shear gradient is zero"
    print("✓ Manual gradient check passed")

    # Numeric gradient check
    param_seq_3d_d = torch.randn(1, 6, dtype=torch.double, requires_grad=True)
    fn_seq_3d = lambda p: transform_3d_point_cloud(x_pc_d, ShearSequential3D.matrix(p))
    assert torch.autograd.gradcheck(fn_seq_3d, (param_seq_3d_d,), eps=1e-6,
                                    atol=1e-4), "Sequential 3D shear gradcheck failed"
    print("✓ Numeric gradient check passed")

    # ------------ Test 5: Full Shear Matrix with 2D ------------
    print("\n5. Testing full shear matrix with 2D images:")
    param_full_2d = torch.randn(1, 2, requires_grad=True)  # 2 parameters for 2D full shear
    matrix_full_2d = ShearFull2D.matrix(param_full_2d)
    res_full_2d = grid_resample(x_img, matrix_full_2d)
    res_full_2d.sum().backward()
    assert param_full_2d.grad is not None, "Full 2D shear gradient is None"
    assert param_full_2d.grad.abs().sum().item() > 0, "Full 2D shear gradient is zero"
    print("✓ Manual gradient check passed")

    # Numeric gradient check
    param_full_2d_d = torch.randn(1, 2, dtype=torch.double, requires_grad=True)
    fn_full_2d = lambda p: grid_resample(x_img_d, ShearFull2D.matrix(p))
    assert torch.autograd.gradcheck(fn_full_2d, (param_full_2d_d,), eps=1e-6,
                                    atol=1e-4), "Full 2D shear gradcheck failed"
    print("✓ Numeric gradient check passed")

    # ------------ Test 6: Full Shear Matrix with 3D ------------
    print("\n6. Testing full shear matrix with 3D point cloud:")
    param_full_3d = torch.randn(1, 6, requires_grad=True)  # 6 parameters for 3D full shear
    matrix_full_3d = ShearFull3D.matrix(param_full_3d)
    out_full_3d = transform_3d_point_cloud(x_pc, matrix_full_3d)
    out_full_3d.sum().backward()
    assert param_full_3d.grad is not None, "Full 3D shear gradient is None"
    assert param_full_3d.grad.abs().sum().item() > 0, "Full 3D shear gradient is zero"
    print("✓ Manual gradient check passed")

    # Numeric gradient check
    param_full_3d_d = torch.randn(1, 6, dtype=torch.double, requires_grad=True)
    fn_full_3d = lambda p: transform_3d_point_cloud(x_pc_d, ShearFull3D.matrix(p))
    assert torch.autograd.gradcheck(fn_full_3d, (param_full_3d_d,), eps=1e-6,
                                    atol=1e-4), "Full 3D shear gradcheck failed"
    print("✓ Numeric gradient check passed")

    # ------------ Test 7: Directed Shear Matrix ------------
    print("\n7. Testing directed shear matrix with images:")
    param_dir = torch.randn(1, 1, requires_grad=True)
    matrix_dir = ShearX2D.matrix(param_dir)
    res_dir = grid_resample(x_img, matrix_dir)
    res_dir.sum().backward()
    assert param_dir.grad is not None, "Directed shear gradient is None"
    assert param_dir.grad.abs().sum().item() > 0, "Directed shear gradient is zero"
    print("✓ Manual gradient check passed")

    # Numeric gradient check
    param_dir_d = torch.randn(1, 1, dtype=torch.double, requires_grad=True)
    fn_dir = lambda p: grid_resample(x_img_d, ShearX2D.matrix(p))
    assert torch.autograd.gradcheck(fn_dir, (param_dir_d,), eps=1e-6, atol=1e-4), "Directed shear gradcheck failed"
    print("✓ Numeric gradient check passed")

    # ------------ Test 8: Orbit Generation for Single Parameter Transforms ------------
    print("\n8. Testing orbit generation for single-parameter transforms_old:")
    orbit_x_2d = ShearX2D.orbit(10, 1.0)
    assert orbit_x_2d is not None, "ShearX2D orbit is None"
    assert orbit_x_2d.shape == (10, 1), f"ShearX2D orbit has incorrect shape: {orbit_x_2d.shape}"

    orbit_y_2d = ShearY2D.orbit(10, 1.0)
    assert orbit_y_2d is not None, "ShearY2D orbit is None"
    assert orbit_y_2d.shape == (10, 1), f"ShearY2D orbit has incorrect shape: {orbit_y_2d.shape}"

    orbit_xy_3d = ShearXY3D.orbit(10, 1.0)
    assert orbit_xy_3d is not None, "ShearXY3D orbit is None"
    assert orbit_xy_3d.shape == (10, 1), f"ShearXY3D orbit has incorrect shape: {orbit_xy_3d.shape}"
    print("✓ Orbit generation tests passed")

    # ------------ Test 9: Parameter Bounds and Projections ------------
    print("\n9. Testing parameter bounds and projections:")
    # Test with a domain of [-1, 1]
    domain = 1.0
    # Create parameters outside the domain
    out_of_bounds = torch.tensor([[1.5, -1.5]], dtype=torch.float32)

    # Test calc_bounds
    lower, upper = Shear2DGeneral.calc_bounds(domain)
    print(f"Domain {domain} → bounds: [{lower}, {upper}]")
    assert torch.allclose(lower, torch.tensor([-1.0, -1.0])), "Incorrect lower bounds"
    assert torch.allclose(upper, torch.tensor([1.0, 1.0])), "Incorrect upper bounds"

    # Test project_parameters with reflection
    reflected = Shear2DGeneral.project_parameters(out_of_bounds, domain, reflect=True)
    print(f"Original: {out_of_bounds}, reflected: {reflected}")
    assert torch.all((reflected >= -1.0) & (reflected <= 1.0)), "Reflection failed to constrain parameters"

    # Test project_parameters with clamping
    clamped = Shear2DGeneral.project_parameters(out_of_bounds, domain, reflect=False)
    print(f"Original: {out_of_bounds}, clamped: {clamped}")
    assert torch.allclose(clamped, torch.tensor([[1.0, -1.0]])), "Clamping failed"

    # Test with single-parameter transform
    single_param = torch.tensor([[2.5]], dtype=torch.float32)
    reflected_single = ShearX2D.project_parameters(single_param, domain, reflect=True)
    print(f"Original: {single_param}, reflected: {reflected_single}")
    assert torch.all((reflected_single >= -1.0) & (reflected_single <= 1.0)), "Single parameter reflection failed"

    print("✓ Parameter bounds and projections tests passed")

    print("\nAll shear gradient and orbit checks passed!")
