import torch

from src.utils.affine_transforms import AffineTransformation2D, AffineTransformations3D
from src.utils.transform_sequence import TransformSequence
from src.utils.transforms.apply import grid_resample, transform_3d_point_cloud, transform_strokes_affine


def get_transformation_sequence_images(name="mnist_default", resample_method=grid_resample, init_method="individual"):
    """
    Return a sequence of transformations for the given name.
    Supports image (2D affine) and point cloud (3D rotation) sequences.
    """
    # map string methods (including 3D & adjusted)
    if not callable(resample_method):
        map_dict = {
            "grid_resample": grid_resample,
            "transform_3d_point_cloud": transform_3d_point_cloud,
            "transform_strokes_affine": transform_strokes_affine,
        }
        if resample_method in map_dict:
            resample_method = map_dict[resample_method]
        else:
            raise ValueError(f"Unknown resample method: {resample_method}")

    # default settings
    neighbour_hood_size = None

    if name in ["mnist_default", "emnist_default"]:
        transformations = [
            AffineTransformation2D.ROTATION.value,
            AffineTransformation2D.SHEARING_X.value,
            AffineTransformation2D.SHEARING_Y.value,
            AffineTransformation2D.SCALING_X.value,
            AffineTransformation2D.SCALING_Y.value
        ]
        domains = [
            (-torch.pi, torch.pi),
            ((-0.5, 0.5),),
            ((-0.5, 0.5),),
            ((1 / 1.3 - 1, 0.3),),
            ((1 / 1.3 - 1, 0.3),)
        ]
        use_individual_param_correction = False
    elif name == "coil_default":
        transformations = [
            AffineTransformation2D.REFLECTION_X.value,
            AffineTransformation2D.ROTATION.value,
            AffineTransformation2D.SHEARING_X.value,
            AffineTransformation2D.SHEARING_Y.value,
            AffineTransformation2D.SCALING_X.value,
            AffineTransformation2D.SCALING_Y.value
        ]
        domains = [
            ((-1.0, 1.0),),
            (-torch.pi, torch.pi),
            ((-0.5, 0.5),),
            ((-0.5, 0.5),),
            ((1 / 1.3 - 1, 0.3),),
            ((1 / 1.3 - 1, 0.3),)
        ]
        use_individual_param_correction = False
    elif name == "si_score_default":
        # rotation only for SI-SCORE (ignore scale/translation variants for now)
        transformations = [AffineTransformation2D.ROTATION.value]
        domains = [(-torch.pi, torch.pi)]
        use_individual_param_correction = False
    elif name == "rotated_mnist_rotation_only":
        transformations = [AffineTransformation2D.ROTATION.value]
        domains = [(-torch.pi, torch.pi)]
        use_individual_param_correction = False
    elif name == "modelnet_default":
        transformations = [AffineTransformations3D.ROTATION.value]
        domains = [(-torch.pi, torch.pi)]
        use_individual_param_correction = False
    elif name in ["biggermnist_default", "bigger_emnist_default"]:
        # Larger canvas allows stronger scaling; user-specified domains
        transformations = [
            AffineTransformation2D.ROTATION.value,
            AffineTransformation2D.SHEARING_X.value,
            AffineTransformation2D.SHEARING_Y.value,
            AffineTransformation2D.SCALING_X.value,
            AffineTransformation2D.SCALING_Y.value
        ]
        domains = [
            (-torch.pi, torch.pi),
            ((-0.5, 0.5),),
            ((-0.5, 0.5),),
            ((1 / (1 + 0.8) - 1, 0.8),),
            ((1 / (1 + 0.8) - 1, 0.8),)
        ]
        use_individual_param_correction = False
    elif name in ["mnist_with_reflection", "mnist_reflection"]:
        # MNIST with horizontal and vertical reflections added
        transformations = [
            AffineTransformation2D.ROTATION.value,
            AffineTransformation2D.REFLECTION_X.value,  # flip over X-axis
            AffineTransformation2D.SHEARING_X.value,
            AffineTransformation2D.SHEARING_X.value,
            AffineTransformation2D.SHEARING_Y.value,
            AffineTransformation2D.SCALING_X.value,
            AffineTransformation2D.SCALING_Y.value,
        ]
        domains = [
            (-torch.pi, torch.pi),
            ((-1.0, 1.0),),  # reflection over X-axis
            ((-0.5, 0.5),),
            ((-0.5, 0.5),),
            ((1 / 1.3 - 1, 0.3),),
            ((1 / 1.3 - 1, 0.3),),
        ]
        use_individual_param_correction = False
    elif name in ["biggermnist_with_reflections", "biggermnist_reflections"]:
        # BiggerMNIST with reflections and stronger scaling range
        transformations = [
            AffineTransformation2D.ROTATION.value,
            AffineTransformation2D.SHEARING_X.value,
            AffineTransformation2D.SHEARING_Y.value,
            AffineTransformation2D.SCALING_X.value,
            AffineTransformation2D.SCALING_Y.value,
            AffineTransformation2D.REFLECTION_X.value,
            AffineTransformation2D.REFLECTION_Y.value,
        ]
        domains = [
            (-torch.pi, torch.pi),
            ((-0.5, 0.5),),
            ((-0.5, 0.5),),
            ((1 / (1 + 0.8) - 1, 0.8),),
            ((1 / (1 + 0.8) - 1, 0.8),),
            ((-1.0, 1.0),),
            ((-1.0, 1.0),),
        ]
        use_individual_param_correction = False
    elif name == "small_affine":
        # Small affine augment: rotation ±30°, scaling ±0.1, shearing ±0.2
        transformations = [
            AffineTransformation2D.ROTATION.value,
            AffineTransformation2D.SHEARING_X.value,
            AffineTransformation2D.SHEARING_Y.value,
            AffineTransformation2D.SCALING_X.value,
            AffineTransformation2D.SCALING_Y.value
        ]
        domains = [
            (-torch.pi / 6, torch.pi / 6),  # ±30 degrees
            ((-0.2, 0.2),),  # shearing x
            ((-0.2, 0.2),),  # shearing y
            ((1 / 1.1 - 1, 0.1),),  # scaling x
            ((1 / 1.1 - 1, 0.1),)  # scaling y
        ]
        use_individual_param_correction = False
    else:
        raise ValueError(f"Unknown transformation sequence name: {name}")

    transform_seq = TransformSequence(
        transformations,
        domains,
        neighbour_hood_size=neighbour_hood_size,
        application_method=resample_method,
        device="cpu",
        reflect=True,
        use_individual_param_correction=use_individual_param_correction,
        init_method=init_method
    )
    return transform_seq
