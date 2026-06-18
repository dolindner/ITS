import math
import torch.nn as nn
from escnn import nn as escnn_nn
from .basic_networks import FlexibleResNet, ESCNNFlexibleResNet, get_flexible_resnet_layer_mapping



def bigger_extended_resnet44(num_classes=47, activation=nn.GELU):
    # stem stride 2 for 56 -> 28
    return FlexibleResNet([32, 64, 128], 7, num_classes=num_classes, in_channels=1, activation=activation, stem_stride=2)


def bigger_extended_equivariant_resnet44(num_classes=47, act_cls=escnn_nn.ReLU, num_rotations=8):
    # automatic size scaling analogous to MNIST equivariant models
    base = [32, 64, 128]
    sizes = [math.ceil(1.2 * s / (num_rotations ** 0.5)) for s in base]
    return ESCNNFlexibleResNet(sizes, 7, num_classes=num_classes, in_channels=1,
                               act_cls=act_cls, rotations=num_rotations, stem_stride=2)



def bigger_extended_resnet_small(num_classes=47, activation=nn.GELU):
    return FlexibleResNet([64, 128, 256,512], 1, num_classes=num_classes, in_channels=1, activation=activation, stem_stride=2)

def _scale_channels_list(channels, divisor):
    return [max(8, math.ceil(c / divisor)) for c in channels]

def bigger_extended_resnet_small_scaled(num_classes=47, activation=nn.GELU, divisor=1):
    base = [64, 128, 256, 512]
    if divisor != 1:
        base = _scale_channels_list(base, divisor)
    return FlexibleResNet(base, 1, num_classes=num_classes, in_channels=1, activation=activation, stem_stride=2)




def get_bigger_extended_mnist_architectures():
    """
    Returns possible bigger EMNIST architectures.
    """
    return [
        "bigger_extended_simple_cnn",
        "bigger_extended_deep_cnn",
        "bigger_extended_resnet44",
        "bigger_extended_resnet_small",

        "bigger_extended_equivariant_resnet44",
        # scaled variants
        "bigger_extended_resnet_small_half",
        "bigger_extended_resnet_small_quarter",
    ]

def get_bigger_extended_mnist_network(architecture, num_classes=47, num_rotations=8):
    """
    Factory function to get specified bigger extended MNIST architecture.
    """
    a = architecture.lower()
    if a == "bigger_extended_resnet44":
        return bigger_extended_resnet44(num_classes=num_classes)
    if a == "bigger_extended_resnet_small":
        return bigger_extended_resnet_small(num_classes=num_classes)
    if a == "bigger_extended_resnet_small_half":
        return bigger_extended_resnet_small_scaled(num_classes=num_classes, divisor=2)
    if a == "bigger_extended_resnet_small_quarter":
        return bigger_extended_resnet_small_scaled(num_classes=num_classes, divisor=4)
    if a == "bigger_extended_equivariant_resnet44":
        return bigger_extended_equivariant_resnet44(num_classes=num_classes, num_rotations=num_rotations)
    raise ValueError(f"Unknown bigger extended MNIST architecture: {architecture}")

def get_bigger_extended_mnist_network_layer(architecture, index, num_classes=47, num_rotations=8):
    """
    Returns (layer_name, capture_mode) for FlexibleResNet architectures.
    """
    a = architecture.lower()
    
    if a == "bigger_extended_resnet44":
        blocks_per_stage = 7
        mapping = get_flexible_resnet_layer_mapping(blocks_per_stage, stages=3)
        return mapping[index]
    elif a == "bigger_extended_resnet_small":
        blocks_per_stage = 1
        mapping = get_flexible_resnet_layer_mapping(blocks_per_stage, stages=4)
        return mapping[index]

    
    raise ValueError(f"Layer mapping not implemented for bigger extended MNIST architecture: {architecture}")