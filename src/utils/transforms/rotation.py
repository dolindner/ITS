
import torch
import math

from src.utils.helper import identity
from src.utils.transforms.base import Transform
from src.utils.transforms.periodic_transform import PeriodicTransform
from src.utils.transforms.bounded_transform import BoundedTransform
from src.utils.transforms.apply import grid_resample, transform_3d_point_cloud

#TODO reprojection function do not generally find the closest solution in matrix space
#This is a problem as converting to quaternion and back may result in a solution that is outside the domain to do there being multiple the same represnation in axis angle represenation(likely to due gimbal lock)
#TODO look at libaries like roma for better matrix and conversion functions




#first 2d cases


class Rotation2D(PeriodicTransform):
    """2D rotation transformation."""
    def __init__(self):
        self.dims = 2

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create a 2D rotation matrix for the given angle parameter.
        
        Args:
            param: Tensor of shape (..., 1) containing rotation angle in radians
            
        Returns:
            Homogeneous rotation matrix of shape (..., 3, 3)
        """
        batch_size = param.shape[:-1]
        # Create a rotation matrix
        rotation_matrix = identity(batch_size, dim=3, dtype=param.dtype, device=param.device)
        # Fill the rotation matrix with the angles
        angle = param.squeeze(-1)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        rotation_matrix[..., 0, 0] = cos_angle
        rotation_matrix[..., 0, 1] = -sin_angle
        rotation_matrix[..., 1, 0] = sin_angle
        rotation_matrix[..., 1, 1] = cos_angle
        return rotation_matrix

    def param_size(self) -> int:
        return 1



    def interval(self):
        """Return the natural interval for this periodic transform."""
        return -math.pi, math.pi


class RotationComplex(Transform):
    """2D rotation using complex numbers."""

    def __init__(self):
        self.dims = 2

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create a 2D rotation matrix using complex number.

        Args:
            param: Tensor of shape (..., 2) with complex components [a, b]

        Returns:
            Homogeneous rotation matrix of shape (..., 3, 3)
        """
        # Normalize to unit complex number
        param = param / torch.norm(param, dim=-1, keepdim=True)

        a = param[..., 0]  # Real part
        b = param[..., 1]  # Imaginary part

        batch_size = param.shape[:-1]

        # Build 2x2 rotation matrix from complex components
        row0 = torch.stack([a, -b], dim=-1)
        row1 = torch.stack([b, a], dim=-1)
        R = torch.stack([row0, row1], dim=-2)

        # Create homogeneous matrix (3x3)
        H = identity(batch_size, 3, dtype=param.dtype, device=param.device)
        H[..., :2, :2] = R
        return H

    def param_size(self) -> int:
        return 2

    def normalize_parameters(self, param: torch.Tensor) -> torch.Tensor:  
        return param / (torch.norm(param, dim=-1, keepdim=True) + 1e-12)

    def normalization_violation(self, param: torch.Tensor) -> torch.Tensor:  
        return torch.abs(torch.norm(param, dim=-1) - 1.0)

    def supports_sobol(self) -> bool:  # CHANGED (was False)
        return True

    def sample_space_param_size(self):  
        return 1  # single uniform -> angle; still accepts 2

    def sobol_to_param(self, sparam: torch.Tensor,domain) -> torch.Tensor:  
        """
        Map Sobol samples in [0,1]^k to unit complex numbers while respecting an angular domain.
          - domain scalar R      -> angle ∈ [-R, R]
          - domain (low, high)   -> angle ∈ [low, high] (supports wrap if low > high)
          - domain (N,2) tensor  -> use first row's [low, high] as angle interval
        Uses first component for angle; returns (...,2) unit complex [cos, sin].
        """
        u = sparam[..., 0].clamp(0.0, 1.0)
        # parse domain to an angle interval
        if domain is None:
            low, high = -math.pi, math.pi
        else:
            dom = torch.as_tensor(domain, dtype=sparam.dtype, device=sparam.device)
            if dom.ndim == 0:
                low, high = -float(abs(dom.item())), float(abs(dom.item()))
            elif dom.ndim == 1 and dom.numel() == 2:
                low, high = float(dom[0].item()), float(dom[1].item())
            elif dom.ndim == 2 and dom.shape[1] == 2:
                low, high = float(dom[0, 0].item()), float(dom[0, 1].item())
            else:
                raise ValueError(f"Unsupported domain for RotationComplex.sobol_to_param: shape={tuple(dom.shape)}")

        # support wrapped arc where high < low
        period = 2.0 * math.pi
        span = (high - low) if high >= low else (high - low) + period
        angle = low + u * span
        # wrap to [-π, π] for a canonical representation
        angle = torch.remainder(angle + math.pi, period) - math.pi

        c = torch.cos(angle)
        s = torch.sin(angle)
        param = torch.stack([c, s], dim=-1)
        return self.normalize_parameters(param)

    def supports_orbit(self) -> bool:
        return True

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        return self.normalize_parameters(param)

    def calc_bounds(self, domain=None, dtype=torch.float32, device="cpu"):
        """
        Interpret domain as an angular interval for θ, returned as tensors of shape (1,):
          - None            -> [-π, π]
          - scalar s        -> [-|s|, |s|]
          - 1D [low, high]  -> [low, high]
          - 2D (N,2)        -> use first row [low, high]
        Note: preserves wrapped domains (low can be > high).
        """
        if domain is None:
            low, high = -math.pi, math.pi
        else:
            dom = torch.as_tensor(domain)
            if dom.ndim == 0:
                a = float(abs(dom.item()))
                low, high = -a, a
            elif dom.ndim == 1 and dom.numel() == 2:
                low, high = float(dom[0].item()), float(dom[1].item())
            elif dom.ndim == 2 and dom.shape[1] == 2:
                low, high = float(dom[0, 0].item()), float(dom[0, 1].item())
            else:
                raise ValueError(f"Unsupported domain for RotationComplex.calc_bounds: shape={tuple(dom.shape)}")
        # Do NOT swap if low > high; this indicates a wrapped arc.
        lower = torch.tensor([low], dtype=dtype, device=device)
        upper = torch.tensor([high], dtype=dtype, device=device)
        return lower, upper

    def sample_param(self, batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """
        Uniformly sample angles θ along the given domain (supports wrapped arcs).
        Maps to unit complex [cosθ, sinθ]. Accepts scalar, 1D [low,high], or 2D (N,2) domains.
        """
        low_t, up_t = self.calc_bounds(domain, dtype=dtype, device=device)  # shape (1,)
        low, high = float(low_t.squeeze(-1).item()), float(up_t.squeeze(-1).item())
        period = 2.0 * math.pi
        span = (high - low) if high >= low else (high - low) + period
        u = torch.rand(batch_size, device=device, dtype=dtype)
        angle = low + u * span
        # wrap angle into [-π, π] (cos/sin is periodic; wrapping is for canonicality)
        angle = torch.remainder(angle + math.pi, period) - math.pi
        return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)

    def default_neighbourhood_size(self, domain=None, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Default neighbourhood for unit complex [cosθ, sinθ].
        - domain is interpreted as an angular interval for θ.
        - If domain is None: return full-circle neighbourhood (diameter = 2.0) per component.
        - If domain is given: return the chord length for the arc span Δ,
          c(Δ) = 2·sin(min(π, Δ)/2), replicated for both components.
        """
        period = 2.0 * math.pi

        if domain is None:
            return torch.full((2,), 2.0, dtype=dtype, device=device)

        dom = torch.as_tensor(domain)
        if dom.ndim == 0:
            a = float(abs(dom.item()))
            low, high = -a, a
        elif dom.ndim == 1 and dom.numel() == 2:
            low, high = float(dom[0].item()), float(dom[1].item())
        elif dom.ndim == 2 and dom.shape[1] == 2:
            low, high = float(dom[0, 0].item()), float(dom[0, 1].item())
        else:
            raise ValueError(f"Unsupported domain for RotationComplex.default_neighbourhood_size: shape={tuple(dom.shape)}")

        if low <= high:
            span = high - low
        else:
            span = (high - low) + period

        step_scalar = 2.0 * math.sin(0.5 * min(math.pi, float(span)))
        step_scalar = max(0.0, min(2.0, step_scalar))
        return torch.full((2,), step_scalar, dtype=dtype, device=device)

    def orbit(self, n_samples: int, domain=2 * math.pi,dim=0, extend: int = 0, shift: int = 0) -> torch.Tensor:
        """Generate an orbit of complex rotation parameters on the unit circle, respecting wrapped domains."""
        # parse bounds (preserve wrap)
        low_t, high_t = self.calc_bounds(domain, dtype=torch.float32, device="cpu")
        low_p, high_p = float(low_t[0].item()), float(high_t[0].item())
        period = 2.0 * math.pi
        eps = 1e-4
        total = n_samples + 2 * extend

        # compute arc length with wrap support
        if low_p > high_p:
            high_mod = high_p + period
            arc_len = high_mod - low_p
        else:
            high_mod = high_p
            arc_len = high_p - low_p

        is_full_circle = (domain is None) or (abs(arc_len - period) < eps)

        if is_full_circle:
            angles = torch.linspace(low_p, low_p + period, total + 1)[:-1]
            if shift:
                angles = angles + shift * (2.0 * math.pi / max(1, n_samples))
        else:
            spacing = arc_len / (n_samples - 1) if n_samples > 1 else 0.0
            start = low_p - extend * spacing
            end = high_mod + extend * spacing
            angles = torch.linspace(start, end, total)
            if shift:
                angles = angles + shift * spacing

        # wrap back to [-π, π]
        angles = torch.remainder(angles + math.pi, period) - math.pi

        # Convert angles to complex numbers on the unit circle
        real_part = torch.cos(angles)
        imag_part = torch.sin(angles)

        # Stack real and imaginary parts to get complex representation [a, b]
        complex_params = torch.stack([real_part, imag_part], dim=-1)

        return complex_params

    def distance(self, param1: torch.Tensor, param2: torch.Tensor) -> torch.Tensor:
        # Normalize parameters to unit circle
        eps = 1e-8  # For numerical stability
        param1_norm = param1 / (torch.norm(param1, dim=-1, keepdim=True) + eps)
        param2_norm = param2 / (torch.norm(param2, dim=-1, keepdim=True) + eps)

        # Compute cosine similarity (dot product)
        cos_theta = torch.sum(param1_norm * param2_norm, dim=-1)

        # Clamp and calculate angular distance
        cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
        return torch.arccos(cos_theta)

    def identity_param(self, batch_size = 1, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Returns the parameter that corresponds to the identity transformation.
        By default, returns a zero tensor.
        Returns:
            A tensor of shape (param_size,) corresponding to the identity transformation.
        """
        return torch.tensor([1.0, 0.0], dtype=dtype, device=device).expand(batch_size, -1)





#3d cases

class Rotation3DEuler(PeriodicTransform):
    """3D rotation using Euler angles."""
    def __init__(self,extrinsic=True):
        super().__init__()
        self.dims = 3
        self.extrinsic = extrinsic  # Use extrinsic ZYX convention (yaw-pitch-roll)

    @staticmethod
    def static_matrix(param: torch.Tensor,extrinsic=True) -> torch.Tensor:
        """
        Create a 3D rotation matrix using extrinsic zyx Euler angles (yaw-pitch-roll).

        Args:
            param: Tensor of shape (..., 3) with rotation angles [yaw, pitch, roll]
                  - yaw (z-axis rotation): horizontal rotation
                  - pitch (y-axis rotation): vertical tilt
                  - roll (x-axis rotation): twist around forward axis

        Returns:
            Homogeneous rotation matrix of shape (..., 4, 4)
        """
        batch_size = param.shape[:-1]

        # Use ZYX convention (yaw-pitch-roll) which is more intuitive
        yaw = param[..., 0]  # rotation around z-axis
        pitch = param[..., 1]  # rotation around y-axis
        roll = param[..., 2]  # rotation around x-axis

        # Create rotation matrices for each axis
        # X-axis rotation (roll)
        rotation_x = torch.zeros(batch_size + (3, 3), dtype=param.dtype, device=param.device)
        rotation_x[..., 0, 0] = 1
        rotation_x[..., 1, 1] = torch.cos(roll)
        rotation_x[..., 1, 2] = -torch.sin(roll)
        rotation_x[..., 2, 1] = torch.sin(roll)
        rotation_x[..., 2, 2] = torch.cos(roll)

        # Y-axis rotation (pitch)
        rotation_y = torch.zeros(batch_size + (3, 3), dtype=param.dtype, device=param.device)
        rotation_y[..., 0, 0] = torch.cos(pitch)
        rotation_y[..., 0, 2] = torch.sin(pitch)
        rotation_y[..., 1, 1] = 1
        rotation_y[..., 2, 0] = -torch.sin(pitch)
        rotation_y[..., 2, 2] = torch.cos(pitch)

        # Z-axis rotation (yaw)
        rotation_z = torch.zeros(batch_size + (3, 3), dtype=param.dtype, device=param.device)
        rotation_z[..., 0, 0] = torch.cos(yaw)
        rotation_z[..., 0, 1] = -torch.sin(yaw)
        rotation_z[..., 1, 0] = torch.sin(yaw)
        rotation_z[..., 1, 1] = torch.cos(yaw)
        rotation_z[..., 2, 2] = 1

        # Apply rotations in ZYX order: Z * Y * X
        # This means: first roll (X), then pitch (Y), then yaw (Z)
        if extrinsic:
            rotation_matrix = torch.matmul(rotation_z, torch.matmul(rotation_y, rotation_x))
        else:
            rotation_matrix = torch.matmul(rotation_x, torch.matmul(rotation_y, rotation_z))

        # Create homogeneous transformation matrix
        id = identity(batch_size, dim=4, dtype=param.dtype, device=param.device)
        id[..., :3, :3] = rotation_matrix
        return id

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create a 3D rotation matrix using Euler angles.
        
        Args:
            param: Tensor of shape (..., 3) with rotation angles around x, y, z axes
            
        Returns:
            Homogeneous rotation matrix of shape (..., 4, 4)
        """
        return self.static_matrix(param)


    def param_size(self) -> int:
        return 3

    def orbit(self, n_samples: int, domain=2*math.pi, extend: int = 0, shift: int = 0) -> None:
        """No orbit for multi-parameter transform."""
        return None

    def interval(self):
        """Return the natural interval for this periodic transform."""
        return -math.pi, math.pi



import torch, math

class Rotation3DEulerUniform(PeriodicTransform):
    """
    3D rotation using Euler angles, with sampling that is uniform over SO(3).
    This replaces the roma-based methods with native quaternion-based implementations.

    ZYX Euler only requiers pitch -pi/2 and pi/2 for a full cover. larger domain should still work note however that uniform sampling would not cover it. Orbit sampling however does so they may be a difference in orbit samlping
    based methods and coordinate descent methods that look at the sampled values. Use individual euler angles or non uniform one if this is not intended.
    """

    def __init__(self, extrinsic=True):
        super().__init__()
        self.dims = 3
        self.extrinsic = extrinsic  # Use extrinsic ZYX (yaw-pitch-roll)


    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """Create a 3D rotation matrix using Euler angles."""
        return Rotation3DEuler.static_matrix(param, self.extrinsic)

    def param_size(self) -> int:
        return 3

    def interval(self):
        """Return the natural interval for this periodic transform."""
        return -math.pi, math.pi

    # ---------------------
    # ---------------------
    @staticmethod
    def _uniform_quat(u: torch.Tensor) -> torch.Tensor:
        """
        Convert [0,1]^3 samples into uniformly distributed quaternions.
        u: Tensor (..., 3)
        Returns quaternion (..., 4) in (x, y, z, w) order.
        """
        u1, u2, u3 = u.unbind(-1)
        q = torch.stack([
            torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2),
            torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2),
            torch.sqrt(u1) * torch.sin(2 * math.pi * u3),
            torch.sqrt(u1) * torch.cos(2 * math.pi * u3),
        ], dim=-1)
        return q  # (x, y, z, w)

    # ---------------------
    # ---------------------
    def _quat_to_euler(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (x, y, z, w) → Euler angles (yaw, pitch, roll) or (xyz)"""
        x, y, z, w = q.unbind(-1)

        if self.extrinsic:
            # Extrinsic ZYX (yaw-pitch-roll)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = torch.atan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * y - z * x)
            sinp = torch.clamp(sinp, -1.0, 1.0)
            pitch = torch.asin(sinp)

            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = torch.atan2(siny_cosp, cosy_cosp)
            angles = torch.stack([yaw, pitch, roll], dim=-1)
        else:
            # Intrinsic XYZ
            sinr_cosp = 2 * (w * x - y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = torch.atan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * y + z * x)
            sinp = torch.clamp(sinp, -1.0, 1.0)
            pitch = torch.asin(sinp)

            siny_cosp = 2 * (w * z - x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = torch.atan2(siny_cosp, cosy_cosp)
            angles = torch.stack([yaw, pitch, roll], dim=-1)

        return angles

    # ---------------------
    # ---------------------
    def sample_param(self, batch_size, domain=None, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """
        Generates uniformly distributed rotations in SO(3) and returns Euler angles.
        """
        u = torch.rand(batch_size, 3, device=device, dtype=dtype)
        q = self._uniform_quat(u)
        euler_angles = self._quat_to_euler(q)
        if self.extrinsic:
            euler_angles[..., 1] *= -1
        return euler_angles

    # ---------------------
    # 🧩 Sobol Sampling
    # ---------------------
    def supports_sobol(self) -> bool:
        return True

    def sample_space_param_size(self) -> int:
        return 3

    def sobol_to_param(self, sparam: torch.Tensor, domain=None) -> torch.Tensor:
        """
        Convert Sobol samples in [0,1]^3 to Euler angles corresponding to
        uniformly distributed rotations in SO(3).
        """
        if sparam.shape[-1] != 3:
            raise ValueError(
                f"Expected Sobol samples of shape (..., 3), got {sparam.shape[-1]}"
            )
        q = self._uniform_quat(sparam)
        euler_angles = self._quat_to_euler(q)
        if self.extrinsic:
            euler_angles[..., 1] *= -1
        return euler_angles



class RotationQuaternion(Transform):
    """3D rotation using quaternions."""
    def __init__(self):
        self.dims = 3
        self.warned = False

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Create a 3D rotation matrix using quaternion.
        
        Args:
            param: Tensor of shape (..., 4) with quaternion components [w, x, y, z]
            
        Returns:
            Homogeneous rotation matrix of shape (..., 4, 4)
        """
        # Normalize quaternion to ensure unit quaternion
        param = param / torch.norm(param, dim=-1, keepdim=True)

        # Expects param with shape (..., 4) in [w, x, y, z] format.
        w = param[..., 0]
        x = param[..., 1]
        y = param[..., 2]
        z = param[..., 3]

        batch_size = param.shape[:-1]
        r00 = 1 - 2 * (y ** 2 + z ** 2)
        r01 = 2 * (x * y - z * w)
        r02 = 2 * (x * z + y * w)
        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x ** 2 + z ** 2)
        r12 = 2 * (y * z - x * w)
        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        r22 = 1 - 2 * (x ** 2 + y ** 2)
        row0 = torch.stack([r00, r01, r02], dim=-1)
        row1 = torch.stack([r10, r11, r12], dim=-1)
        row2 = torch.stack([r20, r21, r22], dim=-1)
        R = torch.stack([row0, row1, row2], dim=-2)
        H = identity(batch_size, dim=4, dtype=param.dtype, device=param.device)
        H[..., :3, :3] = R
        return H

    def param_size(self) -> int:
        return 4

    def normalize_parameters(self, param: torch.Tensor) -> torch.Tensor:
        q = param / (torch.norm(param, dim=-1, keepdim=True) + 1e-12)
        mask = q[..., 0] < 0  # canonicalize (w >= 0)
        return torch.where(mask.unsqueeze(-1), -q, q)

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:  
        return self.normalize_parameters(param)

    def normalization_violation(self, param: torch.Tensor) -> torch.Tensor:  
        return torch.abs(torch.norm(param, dim=-1) - 1.0)

    def supports_sobol(self) -> bool:  # CHANGED (was False)
        return True

    def sample_space_param_size(self):
        # Default to 3 (Shoemake) but accept 4 if user provides (Gaussian).
        return 3

    def sobol_to_param(self, sparam: torch.Tensor,domain) -> torch.Tensor:  
        """
        Map Sobol samples to unit quaternions (canonical w >= 0).
          If last_dim == 3: Shoemake method (u1,u2,u3) -> (x,y,z,w)
          If last_dim == 4: Gaussian method (inverse-normal, normalize)
        Optionally considers per-component bounds by light clamping followed by renormalization.
        """
        d = sparam.shape[-1]
        if d not in (3, 4):
            raise ValueError(f"sobol_to_param expects last dim 3 (Shoemake) or 4 (Gaussian), got {d}")
        if d == 3:
            u1, u2, u3 = sparam.unbind(-1)
            s1 = torch.sqrt(1.0 - u1)
            s2 = torch.sqrt(u1)
            th1 = 2 * math.pi * u2
            th2 = 2 * math.pi * u3
            x = s1 * torch.sin(th1)
            y = s1 * torch.cos(th1)
            z = s2 * torch.sin(th2)
            w = s2 * torch.cos(th2)
            q = torch.stack([w, x, y, z], dim=-1)
        else:  # d == 4 Gaussian normalization method
            eps = 1e-7
            u = sparam.clamp(eps, 1 - eps)
            g = torch.sqrt(torch.tensor(2.0, dtype=u.dtype, device=u.device)) * torch.erfinv(2 * u - 1)
            q = g / (torch.norm(g, dim=-1, keepdim=True) + 1e-12)

        # Canonicalize sign (w >= 0)
        mask = q[..., 0] < 0
        q = torch.where(mask.unsqueeze(-1), -q, q)

        # If a domain is provided, lightly clamp components and renormalize.
        if domain is not None:
            lower, upper = self.calc_bounds(domain, dtype=q.dtype, device=q.device)
            q = torch.clamp(q, lower, upper)
            q = self.normalize_parameters(q)  # re-normalize to unit quaternion

        return q

    def supports_orbit(self) -> bool:  # CHANGED (was False)
        return True


    def distance(self, param1: torch.Tensor, param2: torch.Tensor) -> torch.Tensor:
        # Normalize quaternions and account for double cover
        eps = 1e-8
        param1_norm = param1 / (torch.norm(param1, dim=-1, keepdim=True) + eps)
        param2_norm = param2 / (torch.norm(param2, dim=-1, keepdim=True) + eps)

        # Compute absolute dot product (handles q ≡ -q equivalence)
        dot = torch.abs(torch.sum(param1_norm * param2_norm, dim=-1))

        # Calculate geodesic distance on SO(3)
        dot = torch.clamp(dot, 0.0, 1.0 - eps)  # Avoid acos(>1)
        return 2 * torch.arccos(dot)

    def sample_param(self,batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        q = torch.randn(batch_size, 4, device=device, dtype=dtype)
        return self.normalize_parameters(q)

    def orbit(self,
              n_samples: int,
              domain=1.0,
              dim: int = 0,
              extend: int = 0,
              shift: int = 0) -> torch.Tensor:
        """
        Generate a sequence of quaternions by varying ONE raw component linearly.

        It does NOT correspond to:
          - a constant angular velocity
          - a rotation about a fixed spatial axis
          - uniform sampling on SO(3)

        It is only a diagnostic / visualization tool.

        Args:
            n_samples: number of primary samples
            domain: scalar → [-domain, domain];
                    2-vector (low, high);
                    tensor (k,2) per-component intervals (use row dim)
            dim: which quaternion component (0=w,1=x,2=y,3=z) to vary
            extend: extra samples on each side (still linear in raw component)
            shift: integer shift in sample index (cyclic) after construction
        Returns:
            Tensor (n_samples + 2*extend, 4) of unit quaternions (w >= 0 canonicalized)
        """
        if not self.warned:
            print(
                "Warning: RotationQuaternion.orbit varies a single raw quaternion component; "
                "this is NOT an axis-angle/geodesic rotation path."
            )
            self.warned = True

        if dim < 0 or dim > 3:
            raise ValueError("dim must be in [0,3] for quaternion component selection.")

        # Parse domain into per-component bounds
        if isinstance(domain, (int, float)):
            low = -float(domain)
            high = float(domain)
        else:
            dom_t = torch.as_tensor(domain)
            if dom_t.ndim == 1 and dom_t.numel() == 2:
                low, high = float(dom_t[0]), float(dom_t[1])
            elif dom_t.ndim == 2 and dom_t.shape[1] == 2:
                if dom_t.shape[0] <= dim:
                    raise ValueError("Provided domain rows fewer than required component index.")
                low, high = float(dom_t[dim, 0]), float(dom_t[dim, 1])
            else:
                raise ValueError("Unsupported domain format for quaternion orbit.")

        total = n_samples + 2 * extend
        spacing = (high - low) / (n_samples - 1) if n_samples > 1 else 0.0
        start = low - extend * spacing
        vals = torch.linspace(start, high + extend * spacing, total)
        if shift:
            vals = vals.roll(shifts=shift)

        qs = torch.zeros(total, 4, dtype=vals.dtype)
        qs[:, 0] = 1.0  # start from identity quaternion
        qs[:, dim] = vals

        # Renormalize to unit length
        qs = qs / (qs.norm(dim=-1, keepdim=True) + 1e-12)

        # Canonicalize sign to ensure w >= 0 (avoid double cover flips except when dim==0 crosses 0)
        mask = qs[:, 0] < 0
        qs[mask] = -qs[mask]
        return qs

    def calc_bounds(self, domain=None, dtype=torch.float32, device="cpu"):
        """
        Bounds on unit quaternion components [w, x, y, z].
        Defaults respect canonicalization w ≥ 0:
          - None            -> w ∈ [0,1], others ∈ [-1,1]
          - scalar s        -> w ∈ [0, s], others ∈ [-s, s]
          - 1D [low, high]  -> w ∈ [max(0,low), max(high, max(0,low))], others [low, high]
          - 2D (1,2)        -> same as 1D applied to all
          - 2D (4,2)        -> per‑component; w.low clamped to ≥ 0 and w.high ≥ w.low
        """
        if domain is None:
            lower = torch.tensor([0.0, -1.0, -1.0, -1.0], dtype=dtype, device=device)
            upper = torch.tensor([1.0,  1.0,  1.0,  1.0], dtype=dtype, device=device)
            return lower, upper

        dom = torch.as_tensor(domain, dtype=torch.float64)

        if dom.ndim == 0:
            a = float(abs(dom.item()))
            lower = torch.tensor([0.0, -a, -a, -a], dtype=dtype, device=device)
            upper = torch.tensor([a,   a,  a,  a], dtype=dtype, device=device)
            return lower, upper

        if dom.ndim == 1 and dom.numel() == 2:
            low, high = float(dom[0].item()), float(dom[1].item())
            if low > high:
                low, high = high, low
            w_low = max(0.0, low)
            w_high = max(w_low, high)
            lower = torch.tensor([w_low, low, low, low], dtype=dtype, device=device)
            upper = torch.tensor([w_high, high, high, high], dtype=dtype, device=device)
            return lower, upper

        if dom.ndim == 2 and dom.shape[1] == 2:
            if dom.shape[0] == 1:
                low, high = float(dom[0, 0].item()), float(dom[0, 1].item())
                if low > high:
                    low, high = high, low
                w_low = max(0.0, low)
                w_high = max(w_low, high)
                lower = torch.tensor([w_low, low, low, low], dtype=dtype, device=device)
                upper = torch.tensor([w_high, high, high, high], dtype=dtype, device=device)
                return lower, upper
            if dom.shape[0] == 4:
                lower = dom[:, 0].to(dtype=dtype, device=device).clone()
                upper = dom[:, 1].to(dtype=dtype, device=device).clone()
                # enforce canonicalization on w
                lower[0] = max(lower[0].item(), 0.0)
                upper[0] = max(upper[0].item(), lower[0].item())
                return lower, upper

        raise ValueError(f"Unsupported domain for RotationQuaternion.calc_bounds: shape={tuple(dom.shape)}")

    def default_neighbourhood_size(self, domain=None, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Default neighbourhood for unit quaternions (w,x,y,z):
        - If domain provided: per-component span = upper - lower via calc_bounds.
        - Otherwise: small uniform constant per component.
        Returns a tensor of shape (4,) with per-component neighbourhood sizes.
        """
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
        span = (upper - lower).to(dtype=dtype, device=device)
        return torch.clamp(span, min=1e-8)


    def identity_param(self, batch_size = 1, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Returns the parameter that corresponds to the identity transformation.
        For quaternions, this is [1, 0, 0, 0].
        Returns:
            A tensor of shape (batch_size, 4) corresponding to the identity transformation.
        """
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device).expand(batch_size, -1)

class DirectedRotation3D(PeriodicTransform):
        """Rotation in a specific axis direction in 3D."""
        def __init__(self, axis: int):
            self.dims = 3
            self.axis = axis  # 0: Z, 1: Y, 2: X

        def matrix(self, param: torch.Tensor) -> torch.Tensor:
            """
            Create a 3D rotation matrix that rotates around a specific axis.

            Args:
                param: Tensor of shape (..., 1) with rotation angle

            Returns:
                Homogeneous rotation matrix of shape (..., 4, 4)
            """
            # Create a full parameter vector with zeros except at the specified axis
            expanded_param = torch.zeros(param.shape[:-1] + (3,), dtype=param.dtype, device=param.device)
            expanded_param[..., self.axis] = param.squeeze(-1)

            # Use the Euler rotation matrix function
            return Rotation3DEuler.static_matrix(expanded_param)

        def param_size(self) -> int:
            return 1

        def interval(self):
            """Return the natural interval for this periodic transform."""
            return -math.pi, math.pi



class Rotation2Vec(Transform):
    """3D rotation using 2 vector representation (unbounded)."""

    def __init__(self):
        super().__init__()
        self.dims = 6
        self.warned = False

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        """
        Args:
            param: Tensor of shape (..., 6), split into two 3D vectors.
        Returns:
            Homogeneous (...,4,4) rotation matrices.
        """
        batch = param.shape[:-1]
        a1, a2 = param[..., :3], param[..., 3:6]

        # 1. Normalize the first vector
        b1 = a1 / torch.clamp(a1.norm(dim=-1, keepdim=True), min=1e-8)

        # 2. Project a2 onto b1
        proj = (b1 * a2).sum(-1, keepdim=True) * b1

        # 3. Get the orthogonal component (v2)
        v2 = a2 - proj

        # 4. Normalize the orthogonal component to get b2
        b2 = v2 / torch.clamp(v2.norm(dim=-1, keepdim=True), min=1e-8)

        # 5. Get the third vector
        b3 = torch.cross(b1, b2, dim=-1)

        # Assemble 3×3 and embed into 4×4
        R = torch.stack([b1, b2, b3], dim=-1)  # (...,3,3)


        H = identity(batch, 4, dtype=param.dtype, device=param.device)

        H[..., :3, :3] = R
        return H

    def param_size(self) -> int:
        return 6

    def supports_sobol(self) -> bool:
        return True

    def sample_space_param_size(self):
        """
        For uniform sampling on SO(3), we use quaternion intermediate (3D Shoemake).
        The 2-vector representation itself doesn't naturally support uniform sampling.
        """
        return 3

    def sobol_to_param(self, sparam: torch.Tensor, domain) -> torch.Tensor:
        """
        Map Sobol samples to 2-vector rotation parameters.

        Strategy: Since uniform sampling on SO(3) with 2-vectors is non-trivial,
        we use an intermediate quaternion representation (Shoemake method),
        then convert the resulting rotation to 2-vector form.

        Args:
            sparam: Tensor of shape (..., 3) with uniform samples in [0,1]
            domain: Ignored for rotations (no natural bounds on unit vectors)

        Returns:
            Tensor of shape (..., 6) representing two 3D vectors
        """
        if not self.warned:
            print(
                "Warning: Rotation2Vec uses quaternion intermediate for uniform sampling. "
                "Direct uniform sampling in 2-vector space is non-trivial."
            )
            self.warned = True

        if sparam.shape[-1] != 3:
            raise ValueError(f"sobol_to_param expects last dim 3 (Shoemake), got {sparam.shape[-1]}")

        # Use Shoemake method to get uniform quaternion
        u1, u2, u3 = sparam.unbind(-1)
        s1 = torch.sqrt(1.0 - u1)
        s2 = torch.sqrt(u1)
        th1 = 2 * math.pi * u2
        th2 = 2 * math.pi * u3

        # Quaternion components [w, x, y, z]
        w = s2 * torch.cos(th2)
        x = s1 * torch.sin(th1)
        y = s1 * torch.cos(th1)
        z = s2 * torch.sin(th2)

        # Convert quaternion to rotation matrix (full 3x3)
        r00 = 1 - 2 * (y ** 2 + z ** 2)
        r01 = 2 * (x * y - z * w)
        #r02 = 2 * (x * z + y * w)

        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x ** 2 + z ** 2)
        #r12 = 2 * (y * z - x * w)

        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        #r22 = 1 - 2 * (x ** 2 + y ** 2)


        a1 = torch.stack([r00, r10, r20], dim=-1)
        a2 = torch.stack([r01, r11, r21], dim=-1)

        return torch.cat([a1, a2], dim=-1)

    def sample_param(self, batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """
        Sample random rotation parameters using the 2-vector representation.
        Uses quaternion intermediate for uniform sampling on SO(3).
        """
        # Sample uniform quaternion
        q = torch.randn(batch_size, 4, device=device, dtype=dtype)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)

        # Convert quaternion to rotation matrix
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        r00 = 1 - 2 * (y ** 2 + z ** 2)
        r01 = 2 * (x * y - z * w)
        #r02 = 2 * (x * z + y * w)

        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x ** 2 + z ** 2)
        #r12 = 2 * (y * z - x * w)

        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        #r22 = 1 - 2 * (x ** 2 + y ** 2)

        # Extract first two columns as 2-vectors
        a1 = torch.stack([r00, r10, r20], dim=-1)
        a2 = torch.stack([r01, r11, r21], dim=-1)

        return torch.cat([a1, a2], dim=-1)

    def normalize_parameters(self, param: torch.Tensor) -> torch.Tensor:
        """
        Normalize the 2-vector representation to ensure orthonormality.
        """
        a1, a2 = param[..., :3], param[..., 3:6]

        # Normalize first vector
        b1 = a1 / torch.clamp(a1.norm(dim=-1, keepdim=True), min=1e-8)

        # Gram-Schmidt on second vector
        proj = (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = a2 - proj
        b2 = b2 / torch.clamp(b2.norm(dim=-1, keepdim=True), min=1e-8)

        return torch.cat([b1, b2], dim=-1)

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        """
        For rotation parameters, projection means normalizing to maintain orthonormality.
        Domain and reflect are ignored since rotations have no natural bounds.
        """
        return self.normalize_parameters(param)

    def normalization_violation(self, param: torch.Tensor) -> torch.Tensor:
        """
        Measure how far the parameters are from being orthonormal unit vectors.
        Returns a scalar per batch element.
        """
        a1, a2 = param[..., :3], param[..., 3:6]

        # Check if vectors are unit length
        norm1_violation = torch.abs(a1.norm(dim=-1) - 1.0)
        norm2_violation = torch.abs(a2.norm(dim=-1) - 1.0)

        # Check if vectors are orthogonal
        dot_product = (a1 * a2).sum(dim=-1)
        orthogonal_violation = torch.abs(dot_product)

        # Sum all violations
        return norm1_violation + norm2_violation + orthogonal_violation

    def supports_orbit(self) -> bool:
        return False

    def orbit(self, n_samples: int, domain=None, dim: int = 0, extend: int = 0, shift: int = 0) -> None:
        """
        Orbit is not well-defined for 2-vector representation.
        Use quaternion or axis-angle for orbit generation.
        """
        return None

    def distance(self, param1: torch.Tensor, param2: torch.Tensor) -> torch.Tensor:
        """
        Geodesic distance between rotations on SO(3).
        Convert to rotation matrices and compute angular distance.
        """
        # Convert both to rotation matrices
        R1 = self.matrix(param1)[..., :3, :3]  # Extract 3x3 rotation
        R2 = self.matrix(param2)[..., :3, :3]

        # Compute R1^T @ R2
        R_rel = torch.matmul(R1.transpose(-2, -1), R2)

        # Trace of relative rotation
        trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]

        # Angular distance: arccos((trace - 1) / 2)
        # Clamp to avoid numerical issues with arccos
        cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
        return torch.arccos(cos_angle)

    def default_neighbourhood_size(self, domain=None, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Default neighbourhood size for 2-vector representation.
        Since we're on a manifold, use a small constant per component.
        """
        return torch.full((self.param_size(),), 0.1, dtype=dtype, device=device)

    def calc_bounds(self, domain=None, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return large bounds for the 2-vector representation.
        Since the constraint is orthonormality (not box bounds), we return
        generous bounds that encompass all reasonable unnormalized vectors.
        The actual constraint is enforced via normalize_parameters().

        Args:
            domain: Ignored for rotations (optional scalar for bound magnitude)
            dtype: dtype for output tensors
            device: device for output tensors

        Returns:
            (lower_bounds, upper_bounds) as tensors of shape (6,)
        """
        # Use domain as a scaling factor if provided, otherwise default to large bounds
        if domain is not None:
            dom = torch.as_tensor(domain, dtype=torch.float64)
            if dom.ndim == 0:
                bound = float(abs(dom.item()))
            else:
                bound = 10.0  # Fallback if domain is complex
        else:
            bound = 10.0  # Large default bounds

        lower = torch.full((self.param_size(),), -bound, dtype=dtype, device=device)
        upper = torch.full((self.param_size(),), bound, dtype=dtype, device=device)
        return lower, upper

    def support_calc_bounds(self) -> bool:
        """
        calc_bounds is implemented but returns large values since the real
        constraint is orthonormality, not box bounds.
        """
        return False


    def identity_param(self, batch_size = 1, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Returns the parameter that corresponds to the identity transformation.
        For 2-vector representation, this is two orthonormal vectors aligned with axes.
        Returns:
            A tensor of shape (batch_size, 6) corresponding to the identity transformation.
        """
        identity_vectors = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=dtype, device=device)
        return identity_vectors.expand(batch_size, -1)



# 2d
Rotation_2D = Rotation2D()
RotationComplex2D = RotationComplex()

DirectedRotation2D = Rotation2D
#3d
Rotation3DEulerIn_instance = Rotation3DEuler(extrinsic=False)  # Intrinsic ZYX Euler angles
Rotation3DEuler_instance = Rotation3DEuler()



RotationQuaternion3D = RotationQuaternion()
Rotation2Vec3D = Rotation2Vec()  # 3D rotation using two vectors



RotationZ3D = DirectedRotation3D(0)
RotationY3D = DirectedRotation3D(1)
RotationX3D = DirectedRotation3D(2)


#TODO check if conversion functions are needed.

def euler_to_quaternion(param: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 3D Euler angles (roll, pitch, yaw) to quaternions (w, x, y, z).
    Expects `param` to have shape (..., 3) and returns a tensor of shape (..., 4).
    """
    y,p,r = param.unbind(-1)
    hr, hp, hy = 0.5 * r, 0.5 * p, 0.5 * y
    cr, sr = torch.cos(hr), torch.sin(hr)
    cp, sp = torch.cos(hp), torch.sin(hp)
    cy, sy = torch.cos(hy), torch.sin(hy)

    w  = cr * cp * cy + sr * sp * sy
    x  = sr * cp * cy - cr * sp * sy
    yq = cr * sp * cy + sr * cp * sy
    z  = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, yq, z], dim=-1)

def quaternion_to_euler(param: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions (w, x, y, z) to 3D Euler angles (roll, pitch, yaw).
    Expects `param` to have shape (..., 4) and returns a tensor of shape (..., 3).
    """
    param = param / torch.norm(param, dim=-1, keepdim=True)  # Normalize quaternion
    w, x, yq, z = param.unbind(-1)

    t0 = 2 * (w * x + yq * z)
    t1 = 1 - 2 * (x * x + yq * yq)
    roll = torch.atan2(t0, t1)

    t2 = torch.clamp(2 * (w * yq - z * x), -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = 2 * (w * z + x * yq)
    t4 = 1 - 2 * (yq * yq + z * z)
    yaw = torch.atan2(t3, t4)

    angles= torch.stack([yaw,pitch,roll], dim=-1)
    return angles



def angle_to_complex(param: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 2D rotation angles θ to complex numbers (cos θ, sin θ).
    Expects `param` to have shape (..., 1) and returns a tensor of shape (..., 2).
    """
    θ = param.squeeze(-1)
    return torch.stack([torch.cos(θ), torch.sin(θ)], dim=-1)

def complex_to_angle(param: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 2D complex rotations (cos θ, sin θ) to angles θ.
    Expects `param` to have shape (..., 2) and returns a tensor of shape (..., 1).
    """
    param = param / torch.norm(param, dim=-1, keepdim=True)  # Normalize complex number
    θ = torch.atan2(param[..., 1], param[..., 0])
    return θ.unsqueeze(-1)


def quaternion_to_skew_general(param: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions (w, x, y, z) to skew parameters [p01, p02, p12]
    matching RotationSkew3D.matrix ordering.
    Expects `param` shape (...,4), returns (...,3).
    """
    param = param / torch.norm(param, dim=-1, keepdim=True)  # Normalize quaternion
    w, x, y, z = param.unbind(-1)

    # enforce w >= 0
    mask = w < 0
    w = torch.where(mask, -w, w)
    x = torch.where(mask, -x, x)
    y = torch.where(mask, -y, y)
    z = torch.where(mask, -z, z)

    eps = 1e-7
    half = torch.acos(torch.clamp(w, -1.0 + eps, 1.0 - eps))
    angle = 2.0 * half

    sin_half = torch.sin(half)
    scale = torch.where(
        angle < eps,
        torch.full_like(angle, 2.0),
        angle / sin_half
    )

    # axis-angle vector a = [a1, a2, a3]
    a1 = x * scale
    a2 = y * scale
    a3 = z * scale

    # map to skew params [p01, p02, p12] ugly
    p01 = -a3
    p02 =  a2
    p12 = -a1

    return torch.stack([p01, p02, p12], dim=-1)

def skew_general_to_skew_3d(param: torch.Tensor) -> torch.Tensor:
    """
    Convert skew parameters [p01, p02, p12] to axis-angle parameters [vx, vy, vz].
    Expects `param` shape (...,3), returns (...,3).
    """
    p01, p02, p12 = param.unbind(-1)

    # Reconstruct axis-angle vector a = [a1, a2, a3]
    # mapping: p01 = −a3, p02 = a2, p12 = −a1
    vx = -p12
    vy = p02
    vz = -p01

    return torch.stack([vx, vy, vz], dim=-1)

def skew_3d_to_skew_general(param: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle parameters [vx, vy, vz] to skew parameters [p01, p02, p12].
    Expects `param` shape (...,3), returns (...,3).
    """
    vx, vy, vz = param.unbind(-1)

    # map to skew params [p01, p02, p12] ugly
    p01 = -vz
    p02 =  vy
    p12 = -vx

    return torch.stack([p01, p02, p12], dim=-1)



def skew_general_to_quaternion(param: torch.Tensor) -> torch.Tensor:
    """
    Convert skew‐vector [p01, p02, p12] to quaternion [w, x, y, z],
    matching RotationSkew3D.matrix ordering.
    """
    # Unpack skew params
    p01, p02, p12 = param.unbind(-1)
    # Reconstruct axis-angle vector a = [a1, a2, a3]
    # mapping: p01 = −a3, p02 = a2, p12 = −a1
    vx = -p12
    vy = p02
    vz = -p01
    v = torch.stack([vx, vy, vz], dim=-1)

    # Compute angle and half-angle
    angle = torch.norm(v, dim=-1)
    half = 0.5 * angle
    cos_half = torch.cos(half)
    sin_half = torch.sin(half)

    # Avoid division by zero for small angles
    eps = 1e-7
    sin_half_over_angle = torch.where(
        angle < eps,
        torch.full_like(angle, 0.5),
        sin_half / angle
    )

    # Build quaternion and enforce w>=0
    w = cos_half
    x = v[..., 0] * sin_half_over_angle
    y = v[..., 1] * sin_half_over_angle
    z = v[..., 2] * sin_half_over_angle
    quat = torch.stack([w, x, y, z], dim=-1)

    mask = quat[..., 0] < 0
    quat[mask] = -quat[mask]
    return quat


def quaternion_to_skew_3d(quat: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion [w, x, y, z] to axis-angle parameters [vx, vy, vz].

        This conversion is direct without parameter inversions since we use
        the standard axis-angle to skew-symmetric matrix mapping.

        Args:
            quat: Tensor of shape (..., 4) containing quaternions [w, x, y, z]

        Returns:
            Tensor of shape (..., 3) containing axis-angle parameters [vx, vy, vz]
        """
        w, x, y, z = quat.unbind(-1)

        # Ensure w >= 0 (choose shorter rotation path)
        mask = w < 0
        w = torch.where(mask, -w, w)
        x = torch.where(mask, -x, x)
        y = torch.where(mask, -y, y)
        z = torch.where(mask, -z, z)

        # Convert to axis-angle
        eps = 1e-7
        half_angle = torch.acos(torch.clamp(w, -1.0 + eps, 1.0 - eps))
        angle = 2.0 * half_angle
        sin_half = torch.sin(half_angle)

        # Compute scale factor for axis recovery
        scale = torch.where(
            angle < eps,
            torch.full_like(angle, 2.0),  # Small angle approximation
            angle / sin_half
        )

        # Axis-angle vector (no inversions needed!)
        vx = x * scale
        vy = y * scale
        vz = z * scale

        return torch.stack([vx, vy, vz], dim=-1)

def skew_3d_to_quaternion(param: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle parameters [vx, vy, vz] to quaternion [w, x, y, z].

        Args:
            param: Tensor of shape (..., 3) containing axis-angle parameters

        Returns:
            Tensor of shape (..., 4) containing quaternions [w, x, y, z]
        """
        vx, vy, vz = param.unbind(-1)
        v = torch.stack([vx, vy, vz], dim=-1)

        # Compute angle and half-angle
        angle = torch.norm(v, dim=-1)
        half_angle = 0.5 * angle
        cos_half = torch.cos(half_angle)
        sin_half = torch.sin(half_angle)

        # Avoid division by zero for small angles
        eps = 1e-7
        sin_half_over_angle = torch.where(
            angle < eps,
            torch.full_like(angle, 0.5),  # Small angle approximation
            sin_half / angle
        )

        # Build quaternion
        w = cos_half
        x = vx * sin_half_over_angle
        y = vy * sin_half_over_angle
        z = vz * sin_half_over_angle
        quat = torch.stack([w, x, y, z], dim=-1)

        # Ensure w >= 0 for canonical representation
        mask = quat[..., 0] < 0
        quat[mask] = -quat[mask]

        return quat





if __name__ == "__main__":
    print("\n14. Testing uniform rotation distribution of Rotation3DEulerUniform:")
    with torch.no_grad():
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Generate a large number of samples
        n_samples_uniform = 100000
        uniform_rot_sampler = Rotation3DEulerUniform()

        # Test both sampling methods
        for sample_method_name in ["sample_param", "sobol_to_param"]:
            print(f"  Testing with {sample_method_name}...")
            if sample_method_name == "sample_param":
                params = uniform_rot_sampler.sample_param(n_samples_uniform, device="cpu",domain=None)
            else:  # sobol_to_param
                sobol_engine = torch.quasirandom.SobolEngine(dimension=3)
                sobol_samples = sobol_engine.draw(n_samples_uniform)
                params = uniform_rot_sampler.sobol_to_param(sobol_samples, domain=None)

            # 2. Convert to rotation matrices
            matrices = uniform_rot_sampler.matrix(params)

            # 3. Calculate the trace of the 3x3 rotation part
            # trace(R) = 1 + 2*cos(theta)
            traces = torch.einsum('...ii->...', matrices[..., :3, :3])

            # 4. Calculate the rotation angle theta from the trace
            # theta = acos((trace(R) - 1) / 2)
            cos_theta = (traces - 1.0) / 2.0
            # Clamp values to handle potential floating point inaccuracies
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angles = torch.acos(cos_theta)  # angles are in [0, pi]

            # 5. Plot histogram of angles
            plt.figure(figsize=(10, 6))
            # Plot histogram of sampled angles
            plt.hist(angles.numpy(), bins=100, density=True, label=f'Sampled Distribution ({sample_method_name})')

            # 6. Plot theoretical distribution
            # The PDF for the angle of a uniform rotation is (1/pi) * (1 - cos(theta))
            theta_range = np.linspace(0, np.pi, 200)
            pdf_theoretical = (1.0 / np.pi) * (1.0 - np.cos(theta_range))
            plt.plot(theta_range, pdf_theoretical, 'r-', linewidth=2, label='Theoretical PDF for SO(3)')

            plt.title(f'Distribution of Rotation Angles from {uniform_rot_sampler.__class__.__name__}')
            plt.xlabel('Rotation Angle θ (radians)')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.grid(True)
            plt.show()

            print(
                f"  Plot for {sample_method_name} generated. Visually inspect if the histogram matches the theoretical curve.")




    print("Testing class-based rotation transforms_old...")
    
    # Create test data
    x_img = torch.randn(1, 1, 28, 28)  # 2D image
    x_pc = torch.randn(1, 1024, 3)    # 3D point cloud
    x_img_d = x_img.to(torch.double)  # Double precision for numeric gradient checks
    x_pc_d = x_pc.to(torch.double)

    # Test orbit with different domain formats
    print("\n12. Testing orbit with various domain formats:")

    # Test with scalar domain
    orbit_scalar = Rotation2D.orbit(6, math.pi / 2)
    print(f"Scalar domain orbit shape: {orbit_scalar.shape}")
    assert orbit_scalar.shape == (6, 1), f"Scalar domain orbit has wrong shape: {orbit_scalar.shape}"
    angles_scalar = orbit_scalar.squeeze().numpy()
    range_scalar = angles_scalar.max() - angles_scalar.min()
    assert abs(range_scalar - math.pi) < 1e-5, f"Expected range π/2, got {range_scalar}"
    print(f"Scalar domain orbit covers correct range: {range_scalar:.3f}")

    # Test with vector domain
    domain_vec = torch.tensor([0.0, math.pi])
    orbit_vec = RotationY3D.orbit(5, domain_vec)
    print(f"Vector domain orbit shape: {orbit_vec.shape}")
    angles_vec = orbit_vec.squeeze().numpy()
    assert abs(angles_vec.min() - 0.0) < 1e-5, f"Expected min 0.0, got {angles_vec.min()}"
    assert abs(angles_vec.max() - math.pi) < 1e-5, f"Expected max π, got {angles_vec.max()}"
    print(f"Vector domain orbit covers correct range: {angles_vec.min():.3f} to {angles_vec.max():.3f}")

    # Test with matrix domain
    domain_mat = torch.tensor([[-math.pi / 4, math.pi / 4]])
    orbit_mat = RotationZ3D.orbit(7, domain_mat)
    print(f"Matrix domain orbit shape: {orbit_mat.shape}")
    angles_mat = orbit_mat.squeeze().numpy()
    assert abs(angles_mat.min() - (-math.pi / 4)) < 1e-5, f"Expected min -π/4, got {angles_mat.min()}"
    assert abs(angles_mat.max() - math.pi / 4) < 1e-5, f"Expected max π/4, got {angles_mat.max()}"
    print(f"Matrix domain orbit covers correct range: {angles_mat.min():.3f} to {angles_mat.max():.3f}")

    # Test full circle orbits without duplicating endpoints
    print("\n13. Testing full circle orbit sampling:")
    n_samples = 8
    orbit_full = Rotation2D.orbit(n_samples, math.pi)
    angles_full = orbit_full.squeeze().numpy()

    # Calculate angles from 0 to just under 2π
    expected_angles = torch.linspace(-math.pi, math.pi * (1 - 2 / n_samples), n_samples).numpy()
    assert np.allclose(angles_full, expected_angles), "Full circle orbit not sampled correctly"
    print("Full circle orbit samples without duplicating endpoints")

    # Test 1: 2D Rotation
    print("\n1. Testing 2D rotation:")
    param_2d = torch.tensor([[0.5]], requires_grad=True)
    matrix_2d = Rotation2D.matrix(param_2d)
    out_2d = grid_resample(x_img, matrix_2d)
    out_2d.sum().backward()
    assert param_2d.grad is not None and param_2d.grad.abs().sum() > 0, "2D rotation gradient failed"
    print("2D rotation gradient check passed")
    
    # Test 2: 3D Euler Rotation
    print("\n2. Testing 3D Euler rotation:")
    param_3d_euler = torch.randn(1, 3, requires_grad=True)
    matrix_3d_euler = Rotation3DEuler.matrix(param_3d_euler)
    out_3d_euler = transform_3d_point_cloud(x_pc, matrix_3d_euler)
    out_3d_euler.sum().backward()
    assert param_3d_euler.grad is not None and param_3d_euler.grad.abs().sum() > 0, "3D Euler rotation gradient failed"
    print("3D Euler rotation gradient check passed")
    
    # Test 3: 3D Single-Axis Rotation (X-axis)
    print("\n3. Testing 3D X-axis rotation:")
    param_x = torch.tensor([[0.7]], requires_grad=True)
    matrix_x = RotationX3D.matrix(param_x)
    out_x = transform_3d_point_cloud(x_pc, matrix_x)
    out_x.sum().backward()
    assert param_x.grad is not None and param_x.grad.abs().sum() > 0, "X-axis rotation gradient failed"
    print("X-axis rotation gradient check passed")
    
    # Test 4: 3D Y-axis and Z-axis rotations
    print("\n4. Testing 3D Y-axis and Z-axis rotations:")
    param_y = torch.tensor([[0.6]], requires_grad=True)
    matrix_y = RotationY3D.matrix(param_y)
    out_y = transform_3d_point_cloud(x_pc, matrix_y)
    out_y.sum().backward()
    assert param_y.grad is not None and param_y.grad.abs().sum() > 0, "Y-axis rotation gradient failed"
    print("Y-axis rotation gradient check passed")
    
    param_z = torch.tensor([[0.4]], requires_grad=True)
    matrix_z = RotationZ3D.matrix(param_z)
    out_z = transform_3d_point_cloud(x_pc, matrix_z)
    out_z.sum().backward()
    assert param_z.grad is not None and param_z.grad.abs().sum() > 0, "Z-axis rotation gradient failed"
    print("Z-axis rotation gradient check passed")
    
    
    # Test 6: Quaternion Rotation
    print("\n6. Testing quaternion rotation:")
    param_quat = torch.randn(1, 4, requires_grad=True)
    # Normalize quaternion
    param_quat_norm = param_quat / torch.norm(param_quat, dim=-1, keepdim=True)
    matrix_quat = RotationQuaternion3D.matrix(param_quat_norm)
    out_quat = transform_3d_point_cloud(x_pc, matrix_quat)
    out_quat.sum().backward()
    assert param_quat.grad is not None and param_quat.grad.abs().sum() > 0, "Quaternion rotation gradient failed"
    print("Quaternion rotation gradient check passed")
    
    # Test 7: Complex Number Rotation
    print("\n7. Testing complex number rotation:")
    param_complex = torch.randn(1, 2, requires_grad=True)
    # Normalize complex number
    param_complex_norm = param_complex / torch.norm(param_complex, dim=-1, keepdim=True)
    matrix_complex = RotationComplex2D.matrix(param_complex_norm)
    out_complex = grid_resample(x_img, matrix_complex)
    out_complex.sum().backward()
    assert param_complex.grad is not None and param_complex.grad.abs().sum() > 0, "Complex rotation gradient failed"
    print("Complex rotation gradient check passed")
    
    # Test 8: Parameter Bounds and Projections
    print("\n8. Testing parameter bounds and projections:")
    
    # 2D rotation
    angle_over = torch.tensor([[4.0]])  # Outside of [-π, π]
    angle_proj = Rotation2D.project_parameters(angle_over, 2*math.pi)
    print(f"2D angle {angle_over[0][0]:.3f} projected to {angle_proj[0][0]:.3f}")
    assert -math.pi <= angle_proj[0][0] <= math.pi, "2D angle projection failed"
    
    # 3D Euler rotation
    angles_over = torch.tensor([[4.0, -4.0, 7.0]])  # Outside of [-π, π]
    angles_proj = Rotation3DEuler.project_parameters(angles_over, 2*math.pi)
    print(f"3D Euler angles {angles_over[0].tolist()} projected to {angles_proj[0].tolist()}")
    assert torch.all((angles_proj >= -math.pi) & (angles_proj <= math.pi)), "3D Euler angles projection failed"
    
    # Quaternion normalization
    quat_unnorm = torch.tensor([[2.0, 3.0, 4.0, 5.0]])  # Not normalized
    quat_norm = RotationQuaternion3D.project_parameters(quat_unnorm, None)
    norm = torch.norm(quat_norm, dim=-1)
    print(f"Quaternion normalized to unit norm: {norm.item():.6f}")
    assert torch.allclose(norm, torch.tensor(1.0)), "Quaternion normalization failed"
    
    # Complex number normalization
    complex_unnorm = torch.tensor([[3.0, 4.0]])  # Not normalized
    complex_norm = RotationComplex2D.project_parameters(complex_unnorm, None)
    norm = torch.norm(complex_norm, dim=-1)
    print(f"Complex number normalized to unit norm: {norm.item():.6f}")
    assert torch.allclose(norm, torch.tensor(1.0)), "Complex number normalization failed"
    
    # Test 9: Orbit Generation
    print("\n9. Testing orbit generation:")
    
    # 2D rotation orbit
    orbit_2d = Rotation2D.orbit(10, 2*math.pi)
    print(f"2D rotation orbit shape: {orbit_2d.shape}")
    assert orbit_2d.shape == (10, 1), f"2D rotation orbit has wrong shape: {orbit_2d.shape}"
    
    # Single-axis rotation orbit
    orbit_x = RotationX3D.orbit(8, 2*math.pi)
    print(f"X-axis rotation orbit shape: {orbit_x.shape}")
    assert orbit_x.shape == (8, 1), f"X-axis rotation orbit has wrong shape: {orbit_x.shape}"
    
    # Complex rotation orbit
    orbit_complex = RotationComplex2D.orbit(12, 2*math.pi)
    print(f"Complex rotation orbit shape: {orbit_complex.shape}")
    assert orbit_complex.shape == (12, 2), f"Complex rotation orbit has wrong shape: {orbit_complex.shape}"
    norms = torch.norm(orbit_complex, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms)), "Complex rotation orbit points are not unit norm"

    # Test 10: DirectedRotation2D
    print("\n10. Testing DirectedRotation2D:")
    param_2d_dir = torch.tensor([[0.5]], requires_grad=True)
    matrix_2d_dir = DirectedRotation2D.matrix(param_2d_dir)
    out_2d_dir = grid_resample(x_img, matrix_2d_dir)
    out_2d_dir.sum().backward()
    assert param_2d_dir.grad is not None and param_2d_dir.grad.abs().sum() > 0, "DirectedRotation2D gradient failed"
    print("DirectedRotation2D gradient check passed")


    print("\nAll rotation tests passed!")
