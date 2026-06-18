from torchvision import models
import torch.nn as nn

# Layer mapping for SI-SCORE architectures
SI_LAYER_MAPPINGS = {
    "resnet50_pretrained": {
        0: ("fc", "input"),
        1: ("layer4.2", "output"),
        2: ("layer3.2", "output"),
        3: ("layer2.2", "output"),
        4: ("layer1.2", "output"),
    },
    "vit_b_16_pretrained": {
        0: ("heads.head", "input"),
        1: ("encoder.layers.encoder_layer_11", "output"),  # Last transformer block
        2: ("encoder.layers.encoder_layer_8", "output"),   # 3/4 through transformer
        3: ("encoder.layers.encoder_layer_5", "output"),   # 1/2 through transformer
        4: ("encoder.layers.encoder_layer_2", "output"),   # 1/4 through transformer
        5: ("encoder.dropout", "output"),    # After patch embedding
    },
}

def get_si_network_architectures():
    """
    Returns a list of si-score architectures.
    """
    return ["vit_b_16_pretrained", "resnet50_pretrained"]

def get_si_network(architecture, num_classes=1000, pretrained=True, freeze=True):
    """
    Returns a SI-SCORE architecture based on the specified parameters.
    Supported architectures: "vit_b_16_pretrained", "resnet50_pretrained"
     - pretrained: If True, loads ImageNet pretrained weights. If False, initializes randomly.
     - freeze: If True, freezes all layers except the final classification head.
     - num_classes: Number of output classes for the classification head. Default is 1000 for ImageNet.
     If num_classes is different from 1000, the final layer will be replaced to match num_classes.
    """
    a = architecture.lower()
    if a == "vit_b_16_pretrained":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        if num_classes != 1000:
            in_dim = model.heads.head.in_features
            model.heads.head = nn.Linear(in_dim, num_classes)
        return model
    elif a == "resnet50_pretrained":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        if num_classes != 1000:
            in_dim = model.fc.in_features
            model.fc = nn.Linear(in_dim, num_classes)
        return model
    raise ValueError(f"Unknown SI-SCORE architecture: {architecture}")


def get_si_network_layer(architecture, index, num_classes=1000):
    """
    Returns (layer_name, capture_mode) for the given architecture and index.
    """
    a = architecture.lower()
    if a not in SI_LAYER_MAPPINGS:
        raise ValueError(f"No layer mapping defined for architecture: {architecture}")
    mapping = SI_LAYER_MAPPINGS[a]
    if index not in mapping:
        raise ValueError(f"No layer mapping for index {index} in architecture {architecture}")
    return mapping[index]