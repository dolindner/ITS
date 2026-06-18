import math
import torch.nn as nn
from escnn import nn as escnn_nn
from .basic_networks import FlexibleResNet, ESCNNFlexibleResNet, get_flexible_resnet_layer_mapping



def bigger_resnet44(num_classes=10, activation=nn.GELU):
    return FlexibleResNet([16, 32, 64], 7, num_classes=num_classes, in_channels=1, activation=activation, stem_stride=2)

def bigger_resnetsmall(num_classes=10, activation=nn.GELU):
    return FlexibleResNet([64, 128, 256,512], 1, num_classes=num_classes, in_channels=1, activation=activation, stem_stride=2)

def bigger_resnetmedium(num_classes=10, activation=nn.GELU):
    return FlexibleResNet([64, 128, 256,512], 2, num_classes=num_classes, in_channels=1, activation=activation, stem_stride=2)

def bigger_equivariant_resnet44(num_classes=10, act_cls=escnn_nn.ReLU, num_rotations=8):
    base = [16, 32, 64]
    sizes = [math.ceil(1.2 * s / (num_rotations ** 0.5)) for s in base]
    return ESCNNFlexibleResNet(sizes, 7, num_classes=num_classes, in_channels=1,
                               act_cls=act_cls, rotations=num_rotations, stem_stride=2)

def bigger_equivariant_resnetsmall(num_classes=10, act_cls=escnn_nn.ReLU, num_rotations=8):
    base = [32, 64, 128]
    sizes = [math.ceil(1.2 * s / (num_rotations ** 0.5)) for s in base]
    return ESCNNFlexibleResNet(sizes, 2, num_classes=num_classes, in_channels=1,
                               act_cls=act_cls, rotations=num_rotations, stem_stride=2)

def _scale_channels_list(channels, divisor):
    return [max(8, math.ceil(c / divisor)) for c in channels]

def bigger_resnetsmall_scaled(num_classes=10, activation=nn.GELU, divisor=1):
    base = [64, 128, 256, 512]
    if divisor != 1:
        base = _scale_channels_list(base, divisor)
    return FlexibleResNet(base, 1, num_classes=num_classes, in_channels=1, activation=activation, stem_stride=2)

def get_bigger_mnist_architectures():
    """
    Returns possible architectures.
    """
    return [
        "simple_cnn",
        "deep_cnn",
        "resnet44",
        "resnet_small",
        "resnet_medium",
        "equivariant_resnet44",
        "equivariant_resnet_small",
        # scaled variants for resnet_small
        "resnet_small_half",
        "resnet_small_quarter",
    ]

def get_bigger_mnist_network(architecture, num_classes=10, num_rotations=8):
    """
    Returns a network for the bigger MNIST dataset based on the specified architecture.
    num_rotations is used by requivariant resnets.
    """
    a = architecture.lower()
    if a == "resnet44":
        return bigger_resnet44(num_classes=num_classes)
    if a == "resnet_small":
        return bigger_resnetsmall(num_classes=num_classes)
    if a == "resnet_small_half":
        return bigger_resnetsmall_scaled(num_classes=num_classes, divisor=2)
    if a == "resnet_small_quarter":
        return bigger_resnetsmall_scaled(num_classes=num_classes, divisor=4)
    if a == "resnet_medium":
        return bigger_resnetmedium(num_classes=num_classes)
    if a == "equivariant_resnet44":
        return bigger_equivariant_resnet44(num_classes=num_classes, num_rotations=num_rotations)
    if a == "equivariant_resnet_small":
        return bigger_equivariant_resnetsmall(num_classes=num_classes, num_rotations=num_rotations)
    raise ValueError(f"Unknown bigger MNIST architecture: {architecture}")

def get_bigger_mnist_network_layer(architecture, index, num_classes=10, num_rotations=8):
    """
    Returns (layer_name, capture_mode) for FlexibleResNet architectures.
    """
    a = architecture.lower()
    
    if a == "resnet44":
        blocks_per_stage = 7
        mapping = get_flexible_resnet_layer_mapping(blocks_per_stage, stages=3)
        return mapping[index]
    elif a == "resnet_small":
        blocks_per_stage = 1
        mapping = get_flexible_resnet_layer_mapping(blocks_per_stage, stages=4)
        return mapping[index]
    elif a == "resnet_medium":
        blocks_per_stage = 2
        mapping = get_flexible_resnet_layer_mapping(blocks_per_stage, stages=4)
        return mapping[index]
    
    raise ValueError(f"Layer mapping not implemented for bigger MNIST architecture: {architecture}")