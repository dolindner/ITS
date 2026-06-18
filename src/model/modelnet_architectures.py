# TODO risurfconv does not train. Likely not worth fixing it as pointnet plus pca does even if it only maps to 4 canoncial poses instead of complete invariance.
# PointNet++ based components for ModelNet
try:
    from model.pointnet_plus import PointNetPlus, SAModule, PointNetPlusHalfSized, PointNetPlusQuarterSized
    from src.data.dataset.geometric_wrapper import BatchNormalizeScale, BatchNormalizeScaleEuclidean, \
        TensorGeometricModelWrapper, NormalizeRotationVectorized
    # Add module wrapper to allow usage in nn.Sequential
    from src.data.dataset.geometric_wrapper import NormalizeRotationVectorizedModule

    _POINTNET_AVAILABLE = True
except Exception:
    _POINTNET_AVAILABLE = False

import torch.nn as nn


def _build_pointnetplus(num_classes=10, normalize=True, deterministic_fps=False, smaller=None):
    """
    Builds Pointnet plus architecturs.
    """
    if not _POINTNET_AVAILABLE:
        raise ImportError("PointNetPlus or geometric wrappers not available")
    if smaller is None:
        core = PointNetPlus(num_classes=num_classes)
    elif smaller == 'half':
        core = PointNetPlusHalfSized(num_classes=num_classes)
    elif smaller == 'quarter':
        core = PointNetPlusQuarterSized(num_classes=num_classes)
    if normalize:
        core = nn.Sequential(BatchNormalizeScale(), core)
    model = TensorGeometricModelWrapper(core)
    if deterministic_fps:
        for m in model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
    return model


def _build_pointnetplus_euclidean(num_classes=10, normalize=True, deterministic_fps=False, smaller=None):
    """

    """
    if not _POINTNET_AVAILABLE:
        raise ImportError("PointNetPlus or geometric wrappers not available")
    if smaller is None:
        core = PointNetPlus(num_classes=num_classes)
    elif smaller == 'half':
        core = PointNetPlusHalfSized(num_classes=num_classes)
    elif smaller == 'quarter':
        core = PointNetPlusQuarterSized(num_classes=num_classes)
    if normalize:
        core = nn.Sequential(BatchNormalizeScaleEuclidean(), core)
    model = TensorGeometricModelWrapper(core)
    if deterministic_fps:
        for m in model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
    return model


def _build_pointnetplus_pca(
        num_classes=10,
        normalize=True,
        deterministic_fps=False,
        ensure_proper_rotation=True,
        sort=False,
        max_points=-1,
):
    if not _POINTNET_AVAILABLE:
        raise ImportError("PointNetPlus or geometric wrappers not available")
    core = PointNetPlus(num_classes=num_classes)
    pre = []
    if normalize:
        pre.append(BatchNormalizeScale())
    pre.append(
        NormalizeRotationVectorizedModule(
            max_points=max_points,
            sort=sort,
            ensure_proper_rotation=ensure_proper_rotation,
            randomize=False,
        )
    )
    core = nn.Sequential(*pre, core)
    model = TensorGeometricModelWrapper(core)
    if deterministic_fps:
        for m in model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
    return model


# Change: Norm -> PCA (randomize=True)
def _build_pointnetplus_pca_randomize(
        num_classes=10,
        normalize=True,
        deterministic_fps=False,
        ensure_proper_rotation=True,
        sort=False,
):
    if not _POINTNET_AVAILABLE:
        raise ImportError("PointNetPlus or geometric wrappers not available")
    core = PointNetPlus(num_classes=num_classes)
    pre = []
    if normalize:
        pre.append(BatchNormalizeScale())
    pre.append(
        NormalizeRotationVectorizedModule(
            sort=sort,
            ensure_proper_rotation=ensure_proper_rotation,
            randomize=True,
        )
    )
    core = nn.Sequential(*pre, core)
    model = TensorGeometricModelWrapper(core)
    if deterministic_fps:
        for m in model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
    return model


def _build_pointnetplus_pca_then_norm_randomize(
        num_classes=10,
        normalize=True,
        deterministic_fps=False,
        ensure_proper_rotation=True,
        sort=False,
):
    if not _POINTNET_AVAILABLE:
        raise ImportError("PointNetPlus or geometric wrappers not available")
    core = PointNetPlus(num_classes=num_classes)
    pre = [
        NormalizeRotationVectorizedModule(
            sort=sort,
            ensure_proper_rotation=ensure_proper_rotation,
            randomize=True,
        )
    ]
    if normalize:
        pre.append(BatchNormalizeScale())
    core = nn.Sequential(*pre, core)
    model = TensorGeometricModelWrapper(core)
    if deterministic_fps:
        for m in model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
                print("Set deterministic FPS in SAModule before training")
    return model


def _build_pointnetplus_pca_then_norm_randomize_euclidean(
        num_classes=10,
        normalize=True,
        deterministic_fps=False,
        ensure_proper_rotation=True,
        sort=False,
):
    if not _POINTNET_AVAILABLE:
        raise ImportError("PointNetPlus or geometric wrappers not available")
    core = PointNetPlus(num_classes=num_classes)
    pre = [
        NormalizeRotationVectorizedModule(
            sort=sort,
            ensure_proper_rotation=ensure_proper_rotation,
            randomize=True,
        )
    ]
    if normalize:
        pre.append(BatchNormalizeScaleEuclidean())
    core = nn.Sequential(*pre, core)
    model = TensorGeometricModelWrapper(core)
    if deterministic_fps:
        for m in model.modules():
            if isinstance(m, SAModule):
                m.random_start = False
                print("Set deterministic FPS in SAModule before training")
    return model


def get_modelnet_architectures():
    return [
        "pointnetplus",
        "pointnetplus_pca",
        "pointnetplus_pca_randomize",
        "pointnetplus_pca_then_norm_randomize",
        "pca_randomize",
        "risurfconv",
        "pointnetplus_half",
        "pointnetplus_quarter",
    ]


# Layer mappings for ModelNet architectures
MODELNET_LAYER_MAPPINGS = {
    "pointnetplus": {
        0: ("model.1.mlp.6", "input"),
        1: ("model.1.mlp.4", "input"),
        2: ("model.1.mlp.3", "input"),
        3: ("model.1.mlp.1", "input"),
        4: ("model.1.mlp.0", "input"),
    },
    # PCA then Norm then Core -> core at index 2
    "pointnetplus_pca": {
        0: ("model.2.mlp.6", "input"),
        1: ("model.2.mlp.4", "input"),
        2: ("model.2.mlp.3", "input"),
        3: ("model.2.mlp.1", "input"),
        4: ("model.2.mlp.0", "input"),
    },
    # Identical mapping; sequence is [PCA, Norm, Core] so core is at index 2
    "pointnetplus_pca_randomize": {
        0: ("model.2.mlp.6", "input"),
        1: ("model.2.mlp.4", "input"),
        2: ("model.2.mlp.3", "input"),
        3: ("model.2.mlp.1", "input"),
        4: ("model.2.mlp.0", "input"),
    },
    "pointnetplus_pca_then_norm_randomize": {
        0: ("model.2.mlp.6", "input"),
        1: ("model.2.mlp.4", "input"),
        2: ("model.2.mlp.3", "input"),
        3: ("model.2.mlp.1", "input"),
        4: ("model.2.mlp.0", "input"),
    },
    "risurfconv": {
        0: ("base.classifier", "input"),
        1: ("base.conv4", "output"),
        2: ("base.conv3", "output"),
        3: ("base.conv2", "output"),
        4: ("base.conv1", "output"),
    },
}
