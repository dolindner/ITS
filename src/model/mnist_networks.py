import math

import torch.nn as nn
from escnn import gspaces
from escnn import nn as escnn_nn

from .basic_networks import FlexibleResNet, ESCNNFlexibleResNet, ToGeometric, get_flexible_resnet_layer_mapping


def make_simple_cnn_mnist(num_classes=10, activation=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        activation(),
        nn.Linear(128, num_classes)
    )


def make_equivariant_simple_cnn_mnist(num_classes=10, num_rotations=8, activation=escnn_nn.ReLU):
    r2_act = gspaces.rot2dOnR2(N=num_rotations)
    in_type = escnn_nn.FieldType(r2_act, [r2_act.trivial_repr])
    feat1 = escnn_nn.FieldType(r2_act, 32 * [r2_act.regular_repr])
    feat2 = escnn_nn.FieldType(r2_act, 64 * [r2_act.regular_repr])

    eq_core = escnn_nn.SequentialModule(
        escnn_nn.R2Conv(in_type, feat1, kernel_size=3, padding=1, bias=False),
        activation(feat1, inplace=True),
        escnn_nn.PointwiseAvgPool(feat1, 2),
        escnn_nn.R2Conv(feat1, feat2, kernel_size=3, padding=1, bias=False),
        activation(feat2, inplace=True),
        escnn_nn.PointwiseAvgPool(feat2, 2),
        escnn_nn.GroupPooling(feat2)
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_geo = ToGeometric(in_type)
            self.eq = eq_core
            self.post = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.to_geo(x)
            x = self.eq(x)  # GeometricTensor after group pooling
            x = x.tensor  # ordinary tensor (B, 64, 7, 7)
            return self.post(x)

    return Model()


def make_deep_cnn_mnist(num_classes=10, activation=nn.GELU):
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(128),
        activation(),
        nn.Flatten(),
        nn.Linear(128, num_classes)
    )


def ResNet44(num_classes=10, activation=nn.GELU):
    channels = [16, 32, 64]
    return FlexibleResNet(channels, 7, num_classes=num_classes, in_channels=1, activation=activation)


def ResNetSmall(num_classes=10, activation=nn.GELU):
    return FlexibleResNet([32, 64, 128], 1, num_classes=num_classes, in_channels=1, activation=activation)


def ResNetSmall2(num_classes=10, activation=nn.GELU):
    return FlexibleResNet([64, 128, 256, 512], 1, num_classes=num_classes, in_channels=1, activation=activation,
                          stem_stride=1)


def ResNetMedium(num_classes=10, activation=nn.GELU):
    return FlexibleResNet([32, 64, 128], 2, num_classes=num_classes, in_channels=1, activation=activation)


def EquivariantResNet44(num_classes=10, act_cls=escnn_nn.ReLU, num_rotations=8):
    return ESCNNFlexibleResNet([16, 32, 64], 7, num_classes=num_classes, in_channels=1,
                               act_cls=act_cls, rotations=num_rotations)


def EquivariantResNetSmall(num_classes=10, act_cls=escnn_nn.ReLU, num_rotations=8):
    sizes = [32, 64, 128]
    # multiply by root num_rotations
    sizes = [math.ceil(1.2 * s / (num_rotations ** 0.5)) for s in sizes]

    return ESCNNFlexibleResNet(sizes, 1, num_classes=num_classes, in_channels=1,
                               act_cls=act_cls, rotations=num_rotations)


def EquivariantResNetMedium(num_classes=10, act_cls=escnn_nn.ReLU, num_rotations=8):
    sizes = [32, 64, 128]
    # multiply by root num_rotations
    sizes = [math.ceil(1.2 * s / (num_rotations ** 0.5)) for s in sizes]
    return ESCNNFlexibleResNet(sizes, 2, num_classes=num_classes, in_channels=1,
                               act_cls=act_cls, rotations=num_rotations)


def _scale_channels_list(channels, divisor):
    return [max(8, math.ceil(c / divisor)) for c in channels]


def ResNetSmall_scaled(num_classes=10, activation=nn.GELU, divisor=1):
    base = [64, 128, 256, 512]
    if divisor != 1:
        base = _scale_channels_list(base, divisor)
    return FlexibleResNet(base, 1, num_classes=num_classes, in_channels=1, activation=activation, stem_stride=1)


def get_mnist_architectures():
    return [
        "simple_cnn",
        "equivariant_simple_cnn",
        "deep_cnn",
        "equivariant_deep_cnn",
        "resnet44",
        "equivariant_resnet44",
        "resnetsmall",
        "resnetmedium",
        "equivariant_resnetsmall",
        "equivariant_resnetmedium",
        "gcnn_small",
        "gcnn_medium",
        "z2cnn",
        "p4cnn",
        # new scaled variants for default mapping 'resnet_small'
        "resnet_small_half",
        "resnet_small_quarter",
    ]


def get_mnist_network(architecture, num_classes=10, num_rotations=8):
    """
    Get specific network that outputs num classes. Num rotations is used for equivariant architectures.
    """
    a = architecture.lower()
    if a == "simple_cnn":
        return make_simple_cnn_mnist(num_classes=num_classes)
    if a == "equivariant_simple_cnn":
        return make_equivariant_simple_cnn_mnist(num_classes=num_classes, num_rotations=num_rotations)
    if a == "deep_cnn":
        return make_deep_cnn_mnist(num_classes=num_classes)
    if a == "resnet44":
        return ResNet44(num_classes=num_classes)
    if a == "equivariant_resnet44":
        return EquivariantResNet44(num_classes=num_classes, num_rotations=num_rotations)
    if a == "resnetsmall":
        return ResNetSmall(num_classes=num_classes)
    if a == "resnet_small":
        return ResNetSmall2(num_classes=num_classes)
    if a == "resnet_small_half":
        return ResNetSmall_scaled(num_classes=num_classes, divisor=2)
    if a == "resnet_small_quarter":
        return ResNetSmall_scaled(num_classes=num_classes, divisor=4)
    if a == "resnetmedium":
        return ResNetMedium(num_classes=num_classes)
    if a == "equivariant_resnetsmall":
        return EquivariantResNetSmall(num_classes=num_classes, num_rotations=num_rotations)
    if a == "equivariant_resnetmedium":
        return EquivariantResNetMedium(num_classes=num_classes, num_rotations=num_rotations)
    raise ValueError(f"Unknown MNIST architecture: {architecture}")


def get_mnist_network_layer(architecture, index, num_classes=10, num_rotations=8):
    """
    Returns (layer_name, capture_mode) for FlexibleResNet architectures.
    """
    a = architecture.lower()

    # Define block configurations for each FlexibleResNet architecture
    if a == "resnet44":
        blocks_per_stage = 7  # uniform blocks, 3 stages
        mapping = get_flexible_resnet_layer_mapping(blocks_per_stage, stages=3)
        return mapping[index]
    elif a == "resnet_small":
        blocks_per_stage = 1  # this variant uses 4 stages (ResNetSmall2)
        mapping = get_flexible_resnet_layer_mapping(blocks_per_stage, stages=4)
        return mapping[index]

    raise ValueError(f"Layer mapping not implemented for MNIST architecture: {architecture}")
