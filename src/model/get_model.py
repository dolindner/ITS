from torch import nn

from model.modelnet_architectures import MODELNET_LAYER_MAPPINGS, _build_pointnetplus, _build_pointnetplus_euclidean, \
    _build_pointnetplus_pca_then_norm_randomize_euclidean, _build_pointnetplus_pca, \
    _build_pointnetplus_pca_then_norm_randomize, _build_pointnetplus_pca_randomize, get_modelnet_architectures
from model.tu_berlin_architectures import StrokeAugment, BILSTMSKETCHClassifier, ConvertToOneHotPenState, \
    NormalizeToRangeBatched, NormalizeRotationStrokeBatched, TU_BERLIN_LAYER_MAPPINGS


def get_possible_architectures(dataset_info):
    name = dataset_info.name.lower()
    if name in ["mnist", "rotatedmnist", "rotated_mnist"]:
        from .mnist_networks import get_mnist_architectures
        return get_mnist_architectures()
    if name in ["biggermnist", "bigger_mnist"]:
        from .bigger_mnist_networks import get_bigger_mnist_architectures
        return get_bigger_mnist_architectures()
    # if name == "emnist":
    #    from .extended_mnist_networks import get_extended_mnist_architectures
    #     return get_extended_mnist_architectures()
    if name in ["biggerextendedmnist", "bigger_emnist", "biggeremnist"]:
        from .bigger_extended_mnist_networks import get_bigger_extended_mnist_architectures
        return get_bigger_extended_mnist_architectures()
    # if name == "coil100":
    #    from .coil100_networks import get_coil100_architectures
    #    return get_coil100_architectures()
    if name in ["modelnet", "modelnet10"]:
        return get_modelnet_architectures()
    if name == "si_score":
        from .si_network import get_si_network_architectures
        return get_si_network_architectures()
    if name == "tu_berlin":
        return ["bi_lstm", "bi_lstm_one_hot"]
    raise ValueError(f"Unknown dataset: {dataset_info.name}")


def get_network(dataset_info, architecture, num_classes=None, num_rotations=8):
    name = dataset_info.name.lower()
    num_classes = dataset_info.num_classes if num_classes is None else num_classes
    if name in ["mnist", "rotatedmnist", "rotated_mnist"]:
        from .mnist_networks import get_mnist_network
        return get_mnist_network(architecture, num_classes=num_classes, num_rotations=num_rotations)
    if name in ["biggermnist", "bigger_mnist"]:
        from .bigger_mnist_networks import get_bigger_mnist_network
        return get_bigger_mnist_network(architecture, num_classes=num_classes, num_rotations=num_rotations)
    # if name == "emnist":
    #    from .extended_mnist_networks import get_extended_mnist_network
    #    return get_extended_mnist_network(architecture, num_classes=num_classes, num_rotations=num_rotations)
    if name in ["biggerextendedmnist", "bigger_emnist", "biggeremnist"]:
        from .bigger_extended_mnist_networks import get_bigger_extended_mnist_network
        return get_bigger_extended_mnist_network(architecture, num_classes=num_classes, num_rotations=num_rotations)
    # if name == "coil100":
    #    from .coil100_networks import get_coil100_network
    #    return get_coil100_network(architecture, num_classes=num_classes)
    if name in ["modelnet", "modelnet10"]:
        a = architecture.lower()
        if a == "pointnetplus":
            return _build_pointnetplus(num_classes=num_classes)
        if a == "pointnetplus_euclidean":
            return _build_pointnetplus_euclidean(num_classes=num_classes)
        if a == "pointnetplus_pca_then_norm_randomize_euclidean":
            return _build_pointnetplus_pca_then_norm_randomize_euclidean(num_classes=num_classes, sort=True)
        if a == "pointnetplus_half":
            return _build_pointnetplus(num_classes=num_classes, smaller='half')
        if a == "pointnetplus_quarter":
            return _build_pointnetplus(num_classes=num_classes, smaller='quarter')
        if a == "pointnetplus_pca":
            return _build_pointnetplus_pca(num_classes=num_classes)
        if a in ["pointnetplus_pca_randomize", "pca_randomize", "pca randomize"]:
            return _build_pointnetplus_pca_randomize(num_classes=num_classes)
        if a in ["pointnetplus_pca_then_norm_randomize", "pca_then_norm_randomize"]:
            return _build_pointnetplus_pca_then_norm_randomize(num_classes=num_classes)
        if a in ["pointnetplus_pca_then_norm_randomize_sort", "pca_then_norm_randomize"]:
            return _build_pointnetplus_pca_then_norm_randomize(num_classes=num_classes, sort=True)
        raise ValueError(f"Unknown ModelNet architecture: {architecture}")
    if name == "si_score" or name == "si_score_resnet" or name == "si_score_resnet_no_crop" or name == "si_score_vit_no_crop":
        from .si_network import get_si_network
        return get_si_network(architecture, num_classes=num_classes)
    if name == "tu_berlin":
        a = architecture.lower()
        if a in ("bi_lstm", "bi_lstm_half", "bi_lstm_quarter"):
            # default hidden size 256, halves -> 128, quarters -> 64
            hidden = 256
            if a == "bi_lstm_half":
                hidden = 128
            elif a == "bi_lstm_quarter":
                hidden = 64
            preprocess_module = NormalizeToRangeBatched()
            main = BILSTMSKETCHClassifier(
                input_size=3,
                hidden_size=hidden,
                num_layers=2,
                num_classes=num_classes if num_classes is not None else 250,
                rnn_type='lstm',
                preprocess_module=preprocess_module,
                num_mlp_layers=1,
                dropout=0.5,
                augmentation=StrokeAugment()
            )
            return main
        elif a == "bi_lstm_pca":
            preprocess_module = nn.Sequential(
                NormalizeRotationStrokeBatched(),
                NormalizeToRangeBatched(),
            )
            main = BILSTMSKETCHClassifier(
                input_size=3,
                hidden_size=256,
                num_layers=2,
                num_classes=num_classes if num_classes is not None else 250,
                rnn_type='lstm',
                preprocess_module=preprocess_module,
                num_mlp_layers=1,
                dropout=0.5,
                augmentation=StrokeAugment()
            )
            return main
        elif a in ("bi_lstm_one_hot", "bi_lstm_one_hot_half", "bi_lstm_one_hot_quarter"):
            hidden = 256
            if a == "bi_lstm_one_hot_half":
                hidden = 128
            elif a == "bi_lstm_one_hot_quarter":
                hidden = 64
            preprocess_module = nn.Sequential(
                ConvertToOneHotPenState(),
                NormalizeToRangeBatched()
            )
            main = BILSTMSKETCHClassifier(
                input_size=5,  # (dx, dy, p1, p2, p3)
                hidden_size=hidden,
                num_layers=2,
                num_classes=num_classes if num_classes is not None else 250,
                rnn_type='lstm',
                preprocess_module=preprocess_module,
                num_mlp_layers=1,
                dropout=0.5,
                augmentation=StrokeAugment()
            )
            return main
        raise ValueError(f"Unknown TU-Berlin architecture: {architecture}")
    raise ValueError(f"Unknown dataset: {dataset_info.name}")


def get_network_layer(dataset_info, architecture, index, num_classes=None, num_rotations=8):
    name = dataset_info.name.lower()
    num_classes = dataset_info.num_classes if num_classes is None else num_classes
    if name == "si_score" or name == "si_score_resnet" or name == "si_score_vit_no_crop" or name == "si_score_resnet_no_crop":
        from .si_network import get_si_network_layer
        return get_si_network_layer(architecture, index, num_classes=num_classes)

    # Handle ModelNet architectures
    if name in ["modelnet", "modelnet10"]:
        a = architecture.lower()
        if a not in MODELNET_LAYER_MAPPINGS:
            raise ValueError(f"No layer mapping defined for ModelNet architecture: {architecture}")
        mapping = MODELNET_LAYER_MAPPINGS[a]
        if index not in mapping:
            raise ValueError(f"No layer mapping for index {index} in ModelNet architecture {architecture}")
        return mapping[index]

    # Handle TU Berlin architectures
    if name == "tu_berlin":
        a = architecture.lower()
        if a not in TU_BERLIN_LAYER_MAPPINGS:
            raise ValueError(f"No layer mapping defined for TU Berlin architecture: {architecture}")
        mapping = TU_BERLIN_LAYER_MAPPINGS[a]
        if index not in mapping:
            raise ValueError(f"No layer mapping for index {index} in TU Berlin architecture {architecture}")
        return mapping[index]

    # Handle FlexibleResNet architectures for various datasets
    if name in ["mnist", "rotatedmnist", "rotated_mnist"]:
        from .mnist_networks import get_mnist_network_layer
        return get_mnist_network_layer(architecture, index, num_classes=num_classes, num_rotations=num_rotations)
    if name in ["biggermnist", "bigger_mnist"]:
        from .bigger_mnist_networks import get_bigger_mnist_network_layer
        return get_bigger_mnist_network_layer(architecture, index, num_classes=num_classes, num_rotations=num_rotations)
    # if name == "emnist":
    # from .extended_mnist_networks import get_extended_mnist_network_layer
    # return get_extended_mnist_network_layer(architecture, index, num_classes=num_classes,
    #                                         num_rotations=num_rotations)
    if name in ["biggerextendedmnist", "bigger_emnist", "biggeremnist"]:
        from .bigger_extended_mnist_networks import get_bigger_extended_mnist_network_layer
        return get_bigger_extended_mnist_network_layer(architecture, index, num_classes=num_classes,
                                                       num_rotations=num_rotations)
    # if name == "coil100":
    # from .coil100_networks import get_coil100_network_layer
    # return get_coil100_network_layer(architecture, index, num_classes=num_classes)

    raise ValueError(f"get_network_layer not implemented for dataset: {dataset_info.name}")


def get_max_layer_index(dataset_info, architecture, num_classes=None, num_rotations=8) -> int:
    """
    Tries to find the maximum valid layer index by calling get_network_layer until it fails.
    Returns the last successful index.
    """
    index = 0
    while True:
        try:
            # We only need to check if it runs without error
            get_network_layer(dataset_info, architecture, index, num_classes=num_classes, num_rotations=num_rotations)
            index += 1
        except (ValueError, KeyError, IndexError):
            # These exceptions are typically raised for an invalid index.
            # The last valid index was `index - 1`.
            return index - 1
