import math

import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as escnn_nn

from model.pointnet_plus import SAModule
from .rot_resnet import ESCNNFlexibleResNet


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, activation=nn.GELU):
        """
        A standard ResNet basic block with two 3x3 convolutions and a possible projection for the shortcut.
        The activation function is applied after each convolution and at the end of the block.

        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
            stride: Stride for the first convolution (used for downsampling)
            activation: Activation function class to use (e.g., nn.ReLU, nn.GELU)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.act = activation()

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        shortcut = x if self.proj is None else self.proj(x)
        return self.act(y + shortcut)


class FlexibleResNet(nn.Module):
    def __init__(self, channels, blocks_per_stage, num_classes=10, in_channels=1, activation=nn.ReLU, stem_stride=1):
        """
        A flexible ResNet implementation where the number of channels and blocks per stage can be specified.
        Args:
            channels: List of output channels for each stage (not including the stem)
            blocks_per_stage: List of number of blocks for each stage (or int for uniform blocks)
            num_classes: Number of output classes for the final classifier
            in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
            activation: Activation function class to use (e.g., nn.ReLU, nn.GELU)
            stem_stride: Stride for the initial convolutional stem (default 1, set to 2 for more downsampling)
        """
        super().__init__()
        if isinstance(blocks_per_stage, int):
            blocks_per_stage = [blocks_per_stage] * len(channels)
        if len(channels) != len(blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must have same length")
        self.stem_conv = nn.Conv2d(in_channels, channels[0], 3, stride=stem_stride, padding=1, bias=False)  # changed
        self.stem_bn = nn.BatchNorm2d(channels[0])
        self.stem_act = activation()

        stages = []
        in_ch = channels[0]
        for idx, (out_ch, n) in enumerate(zip(channels, blocks_per_stage)):
            stride = 1 if idx == 0 else 2
            blocks = [BasicBlock(in_ch, out_ch, stride=stride, activation=activation)]
            in_ch = out_ch
            for _ in range(n - 1):
                blocks.append(BasicBlock(in_ch, in_ch, stride=1, activation=activation))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        for stage in self.stages:
            x = stage(x)
        x = self.head_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def convert_flexible_resnet_to_dropout_sequential(model: FlexibleResNet, dropout_p: float = 0.5) -> nn.Sequential:
    """
    Converts a FlexibleResNet to a nn.Sequential model with a dropout layer
    after the flatten operation.
    """
    if not isinstance(model, FlexibleResNet):
        raise TypeError("Input model must be an instance of FlexibleResNet")

    layers = [
        model.stem_conv,
        model.stem_bn,
        model.stem_act,
        *model.stages,
        model.head_pool,
        nn.Flatten(1),
        nn.Dropout(p=dropout_p),
        model.fc
    ]
    return nn.Sequential(*layers)


def convert_flexible_resnet_to_sequential(model: FlexibleResNet) -> nn.Sequential:
    """
    Converts a FlexibleResNet to a nn.Sequential model.
    """
    if not isinstance(model, FlexibleResNet):
        raise TypeError("Input model must be an instance of FlexibleResNet")

    layers = [
        model.stem_conv,
        model.stem_bn,
        model.stem_act,
        *model.stages,
        model.head_pool,
        nn.Flatten(1),
        model.fc
    ]
    return nn.Sequential(*layers)


class ToGeometric(nn.Module):
    """
    Wraps a tensor in a GeometricTensor with the specified field type.
     If the input is already a GeometricTensor, it returns it unchanged.
    """

    def __init__(self, field_type: escnn_nn.FieldType):
        super().__init__()
        self.field_type = field_type

    def forward(self, x: torch.Tensor):
        if isinstance(x, escnn_nn.GeometricTensor):
            return x  # already geometric
        return escnn_nn.GeometricTensor(x, self.field_type)


class FromGeometric(nn.Module):
    def forward(self, x):
        return x.tensor if isinstance(x, escnn_nn.GeometricTensor) else x


class P4CNN(nn.Module):
    def __init__(self, num_classes=10, activation=escnn_nn.ReLU):
        super().__init__()
        r2_act = gspaces.rot2dOnR2(N=8)

        in_type = escnn_nn.FieldType(r2_act, [r2_act.trivial_repr])
        feat_type = escnn_nn.FieldType(r2_act, 20 * [r2_act.regular_repr])

        self.conv1 = escnn_nn.R2Conv(in_type, feat_type, kernel_size=3, padding=1, bias=False)
        self.bn1 = escnn_nn.InnerBatchNorm(feat_type)
        self.conv2 = escnn_nn.R2Conv(feat_type, feat_type, kernel_size=3, padding=1, bias=False)
        self.bn2 = escnn_nn.InnerBatchNorm(feat_type)
        self.conv3 = escnn_nn.R2Conv(feat_type, feat_type, kernel_size=7, bias=False)
        self.bn3 = escnn_nn.InnerBatchNorm(feat_type)

        # switched Max -> Avg for equivariant pooling
        self.pool1 = escnn_nn.PointwiseAvgPool(feat_type, 2)
        self.pool2 = escnn_nn.PointwiseAvgPool(feat_type, 2)

        # Proper equivariant activation (needs field type)
        self.act = activation(feat_type, inplace=True)

        self.gpool = escnn_nn.GroupPooling(feat_type)

        self.fc1 = nn.Linear(20, 50)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = escnn_nn.GeometricTensor(x, self.conv1.in_type)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.gpool(x)  # GeometricTensor with trivial reps
        x = x.tensor.view(x.tensor.size(0), -1)  # (B,20)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def get_flexible_resnet_layer_mapping(blocks_per_stage, stages):
    """
    Generate layer mapping for FlexibleResNet based on its structure.
    Index 0 is always the final classifier 'fc'.
    Higher indices map to blocks in reverse order (later blocks have lower indices).
    
    Args:
        blocks_per_stage: List of number of blocks per stage or int for uniform blocks
    
    Returns:
        Dict mapping indices to (layer_name, capture_mode) tuples
    """
    if isinstance(blocks_per_stage, int):
        # If uniform blocks, we need to know how many stages - assume 3 stages for common cases
        blocks_per_stage = [blocks_per_stage] * stages

    mapping = {0: ("fc", "input")}

    index = 1
    # Go through stages in reverse order (last stage first)
    for stage_idx in reversed(range(len(blocks_per_stage))):
        num_blocks = blocks_per_stage[stage_idx]
        # Go through blocks in reverse order within each stage (last block first)
        for block_idx in reversed(range(num_blocks)):
            layer_name = f"stages.{stage_idx}.{block_idx}.act"
            mapping[index] = (layer_name, "input")
            index += 1

    return mapping


def get_encoder_for_resnet(model, dim=512, vae=False):
    """
    Creates an encoder and decoder for a FlexibleResNet model.
    The encoder compresses the input to a latent space of dimension 'dim'.
    The decoder reconstructs the input from the latent space.
    For VAE, the encoder returns (mu, log_var) for reparameterization.
    Models are scaled down to approximate half the FLOPs of the original model.
    """
    # More robust check that works even with reloaded modules
    if not (isinstance(model, FlexibleResNet) or
            type(model).__name__ == 'FlexibleResNet' or
            hasattr(model, 'stem_conv') and hasattr(model, 'stages')):
        raise ValueError(f"Model must be an instance of FlexibleResNet, got {type(model)}")

    # Extract model structure
    in_channels = model.stem_conv.in_channels
    stem_stride = model.stem_conv.stride[0]
    activation = type(model.stem_act)

    # Get channels for stem and each stage
    stem_channels = model.stem_conv.out_channels
    stage_channels = []
    for stage in model.stages:
        # The out_channels for a stage is the out_channels of its first block's final conv
        stage_channels.append(
            stage[0].conv3.out_channels if hasattr(stage[0], 'conv3') else stage[0].conv2.out_channels)

    blocks_per_stage = [len(stage) for stage in model.stages]

    # Scale down for half FLOPs: divide channels by sqrt(2)
    # FLOPs ~ channels^2, so halving FLOPs means dividing channels by sqrt(2)
    scale_factor = math.sqrt(2)

    # Scale down channels
    encoder_stem_channels = max(8, int(stem_channels / scale_factor))
    encoder_stage_channels = [max(8, int(ch / scale_factor)) for ch in stage_channels]

    # FlexibleResNet expects channels to be ONLY the stage channels, not including stem
    # The stem channels are set separately via the first element or handled internally
    # Let's check if channels should include stem or not by looking at original model
    encoder_channels = encoder_stage_channels  # Try without stem first

    # Also reduce depth slightly
    # blocks_per_stage should have same length as channels list
    encoder_blocks = [max(1, b // 2) for b in blocks_per_stage]

    # Encoder: Smaller ResNet-like network
    encoder = FlexibleResNet(
        channels=encoder_channels,
        blocks_per_stage=encoder_blocks,
        num_classes=dim,
        in_channels=in_channels,
        activation=activation,
        stem_stride=stem_stride
    )

    # The feature extractor part of the encoder (everything except final FC)
    feature_extractor = nn.Sequential(
        encoder.stem_conv,
        encoder.stem_bn,
        encoder.stem_act,
        *encoder.stages,
        encoder.head_pool,
        nn.Flatten(1)
    )

    # Calculate the actual output dimension after pooling
    with torch.no_grad():
        dummy_input = torch.zeros(1, in_channels, 32, 32)  # Assume 32x32 input
        final_feature_dim = feature_extractor(dummy_input).shape[1]

    if vae:
        # For VAE, add two heads: mu and log_var
        mu_head = nn.Linear(final_feature_dim, dim)
        log_var_head = nn.Linear(final_feature_dim, dim)

        class VAEEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = feature_extractor
                self.mu_head = mu_head
                self.log_var_head = log_var_head

            def forward(self, x):
                features = self.feature_extractor(x)
                return self.mu_head(features), self.log_var_head(features)

        encoder_model = VAEEncoder()
    else:
        # Standard encoder: single head
        fc = nn.Linear(final_feature_dim, dim)
        encoder_model = nn.Sequential(feature_extractor, fc)

    # Decoder: Reverse architecture with transposed convolutions
    decoder_layers = []

    # Calculate spatial size after encoding (depends on stride and pooling)
    total_stride = stem_stride * (2 ** len(model.stages))  # Assuming stride 2 per stage
    initial_spatial_size = 32 // total_stride  # Assuming 32x32 input
    initial_spatial_size = max(1, initial_spatial_size)

    # Start with linear layer to expand latent to spatial feature map
    initial_features = encoder_stage_channels[-1]
    decoder_layers.append(nn.Linear(dim, initial_features * initial_spatial_size * initial_spatial_size))
    decoder_layers.append(nn.Unflatten(1, (initial_features, initial_spatial_size, initial_spatial_size)))

    # Reverse the encoder stages
    decoder_stage_channels = encoder_stage_channels[::-1]

    # Build upsampling stages (reverse of encoder stages)
    for i in range(len(decoder_stage_channels) - 1):
        in_ch = decoder_stage_channels[i]
        out_ch = decoder_stage_channels[i + 1]

        # Upsample by 2x (reverse of stride 2 downsampling in encoder)
        decoder_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.BatchNorm2d(out_ch))
        decoder_layers.append(activation())

    # Final upsampling to match stem stride
    final_ch = encoder_stem_channels
    if len(decoder_stage_channels) > 0:
        decoder_layers.append(
            nn.ConvTranspose2d(decoder_stage_channels[-1], final_ch, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.BatchNorm2d(final_ch))
        decoder_layers.append(activation())

    # Final conv to get back to input channels (no stride change)
    if stem_stride > 1:
        decoder_layers.append(nn.ConvTranspose2d(final_ch, in_channels, kernel_size=stem_stride * 2, stride=stem_stride,
                                                 padding=stem_stride // 2))
    else:
        decoder_layers.append(nn.Conv2d(final_ch, in_channels, kernel_size=3, padding=1))

    # Optional: Add sigmoid/tanh activation for image reconstruction
    # decoder_layers.append(nn.Sigmoid())  # Uncomment if images are normalized to [0, 1]

    decoder = nn.Sequential(*decoder_layers)

    return encoder_model, decoder


def make_deterministic(model, random=False, verbose=True):
    """
    Disables randomness for SAModule
    """
    model.aug = False
    for module in model.modules():
        if isinstance(module, SAModule):
            module.random_start = random
            if verbose:
                print(f"Set random_start={random} for {module.__class__.__name__}")


def find_last_linear_layer(model: torch.nn.Module) -> torch.nn.Linear:
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Linear):
            return m
    raise ValueError("No nn.Linear layer found in the model")


class PreActBlock(nn.Module):
    """
    Wraps a BasicBlock to return the pre-activation sum (y + shortcut)
    i.e. omits the final self.act(...) so a downstream head can apply it.
    """

    def __init__(self, block: BasicBlock):
        super().__init__()
        # reuse convolutional / BN / proj modules from the original block
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.proj = block.proj
        # keep a local pre-activation (same type as original block.act)
        self._pre_act = type(block.act)()

    def forward(self, x):
        y = self._pre_act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        shortcut = x if self.proj is None else self.proj(x)
        return y + shortcut


def split_flexible_resnet_for_ash(model, split_pos=0):
    """
    Split FlexibleResNet so ASH runs before the final block activation.
    Assumes every block exposes `act` and moves the last block's activation
    into the head by replacing the block with `PreActBlock`.
    """
    if not isinstance(model, FlexibleResNet):
        raise ValueError("Model must be an instance of FlexibleResNet")

    num_stages = len(model.stages)
    if split_pos < 0 or split_pos > num_stages + 1:
        raise ValueError(f"split_pos must be between 0 and {num_stages + 1}")

    # split_pos == 0: full backbone (after pool)
    if split_pos == 0:
        backbone = nn.Sequential(
            model.stem_conv,
            model.stem_bn,
            model.stem_act,
            *model.stages,
            model.head_pool
        )
        head = nn.Sequential(nn.Flatten(1), model.fc)
        return backbone, head

    # Determine how many stages to keep in backbone
    stages_to_backbone = num_stages - split_pos + 1
    backbone_stages = [s for s in model.stages[:stages_to_backbone]]
    head_stages = [s for s in model.stages[stages_to_backbone:]]

    # If no stages in backbone, move stem_act into head
    if len(backbone_stages) == 0:
        backbone = nn.Sequential(model.stem_conv, model.stem_bn)
        head = nn.Sequential(
            model.stem_act,
            *head_stages,
            model.head_pool,
            nn.Flatten(1),
            model.fc
        )
        return backbone, head

    # Replace the final block in the last backbone stage with PreActBlock
    last_stage = backbone_stages[-1]
    last_block = last_stage[-1]
    last_stage_blocks = list(last_stage)
    last_stage_blocks[-1] = PreActBlock(last_block)
    backbone_stages[-1] = nn.Sequential(*last_stage_blocks)

    # Post-activation (same class as the original block.act) goes to the head
    post_act = type(last_block.act)()

    head = nn.Sequential(post_act, *head_stages, model.head_pool, nn.Flatten(1), model.fc)
    backbone = nn.Sequential(model.stem_conv, model.stem_bn, model.stem_act, *backbone_stages)
    return backbone, head


def get_max_split_pos_for_flexible_resnet(model):
    """
    Returns the maximum valid split_pos for a FlexibleResNet (len(stages) + 1).
    """
    if not isinstance(model, FlexibleResNet):
        raise ValueError("Model must be an instance of FlexibleResNet")
    return len(model.stages) + 1


def compare_model_and_split(model: FlexibleResNet, split_pos: int, atol=1e-6, rtol=1e-5):
    model.eval()
    backbone, head = split_flexible_resnet_for_ash(model, split_pos)

    # Put split parts in eval mode
    backbone.eval()
    head.eval()
    print(backbone)

    # deterministic input
    torch.manual_seed(0)
    x = torch.randn(2, model.stem_conv.in_channels, 32, 32)

    with torch.no_grad():
        orig_out = model(x)
        mid = backbone(x)
        # If backbone returns a GeometricTensor or non-tensor, this test expects plain tensors (FlexibleResNet does)
        split_out = head(mid)

    if not torch.allclose(orig_out, split_out, atol=atol, rtol=rtol):
        diff = (orig_out - split_out).abs()
        max_diff = diff.max().item()
        return False, max_diff
    return True, 0.0


def convert_flexible_resnet_to_gcnn(
        model: FlexibleResNet,
        num_rotations: int = 8,
        reflection: bool = False,
        activation=torch.nn.GELU,
        enable_auto_padding: bool = False
) -> 'GroupResNet':
    """
    Converts a FlexibleResNet model to a GroupResNet with matching architecture.

    Args:
        model: FlexibleResNet instance to convert
        num_rotations: Number of rotations in the group
        reflection: If True, use dihedral group (rotations + reflections)
        activation: Activation function class

    Returns:
        GroupResNet instance with scaled channels to match parameter count
    """
    if not isinstance(model, FlexibleResNet):
        raise TypeError("Input model must be an instance of FlexibleResNet")

    # Extract structure from ResNet
    in_channels = model.stem_conv.in_channels
    num_classes = model.fc.out_features
    stem_stride = model.stem_conv.stride[0]

    # Get channel sizes from stages
    stage_channels = []
    for stage in model.stages:
        first_block = stage[0]
        if hasattr(first_block, 'conv3'):
            out_ch = first_block.conv3.out_channels
        else:
            out_ch = first_block.conv2.out_channels
        stage_channels.append(out_ch)

    # Get blocks per stage
    blocks_per_stage = [len(stage) for stage in model.stages]

    # Scale channels to maintain similar parameter count
    # Group convolutions multiply parameters by group size
    # For dihedral groups, the group size is 2 * num_rotations
    group_size = 2 * num_rotations if reflection else num_rotations
    scale_factor = math.sqrt(group_size)
    scaled_channels = [max(8, int(ch / scale_factor)) for ch in stage_channels]

    # Import here to avoid circular dependency
    from .rot_resnet import GroupResNet

    return GroupResNet(
        channels=scaled_channels,
        blocks_per_stage=blocks_per_stage,
        num_classes=num_classes,
        in_channels=in_channels,
        activation=activation,
        num_rotations=num_rotations,
        use_reflection=reflection,
        stem_stride=stem_stride,
        pad_blocks=enable_auto_padding
    )


def convert_flexible_resnet_to_escnn(
        model: FlexibleResNet,
        rotations: int = 8,
        reflection: bool = False,
        continuous: bool = False,
        max_frequency: int = None,
        act_cls=escnn_nn.ReLU,
        enable_auto_padding: bool = False,
        pad_input: bool = False
) -> ESCNNFlexibleResNet:
    """
    Converts a FlexibleResNet model to an ESCNNFlexibleResNet with matching architecture.

    CRITICAL INSIGHT: ESCNN uses steerable convolutions where the number of parameters
    is determined by the kernel basis size, NOT by naive channel multiplication.

    For regular representations (discrete groups):
    - Kernel basis size is approximately constant per field pair
    - Total params ≈ fields_in * fields_out * basis_size
    - Basis_size is roughly independent of group size!
    - So we need to INCREASE fields to compensate for reduced total channels

    For continuous groups:
    - Constraints are even stronger, fewer basis elements
    - Need even MORE fields to match parameter count

    Args:
        model: FlexibleResNet instance to convert
        rotations: Number of rotations (for discrete groups)
        reflection: If True, use dihedral/O(2) group instead of cyclic/SO(2)
        continuous: If True, use continuous groups (SO(2) or O(2))
        max_frequency: Maximum frequency for continuous groups (defaults to rotations//2)
        act_cls: ESCNN activation function class

    Returns:
        ESCNNFlexibleResNet instance with scaled channels to match parameter count
    """
    if not isinstance(model, FlexibleResNet):
        raise TypeError("Input model must be an instance of FlexibleResNet")

    # Extract structure from ResNet
    in_channels = model.stem_conv.in_channels
    num_classes = model.fc.out_features
    stem_stride = model.stem_conv.stride[0]

    # Get channel sizes from stages
    stage_channels = []
    for stage in model.stages:
        first_block = stage[0]
        if hasattr(first_block, 'conv3'):
            out_ch = first_block.conv3.out_channels
        else:
            out_ch = first_block.conv2.out_channels
        stage_channels.append(out_ch)

    # Get blocks per stage
    blocks_per_stage = [len(stage) for stage in model.stages]

    # Calculate scaling based on kernel basis size
    # For a 3x3 kernel, standard conv has 9 * C_in * C_out parameters
    # For ESCNN with regular repr, basis size is roughly constant (around 9-20 depending on group)
    # So: standard params ≈ 9 * C_in * C_out
    #     ESCNN params ≈ basis_size * fields_in * fields_out
    # To match: fields ≈ C * sqrt(9 / basis_size)

    if continuous:
        if max_frequency is None:
            max_frequency = rotations // 2

        scale_factor = 11 / 9
        if reflection:
            scale_factor *= math.sqrt(2)
    else:
        # Discrete groups: regular representation
        group_size = 2 * rotations if reflection else rotations

        scale_factor = 1.222222222 / math.sqrt(group_size)

    scaled_channels = [max(1, int(ch * scale_factor)) for ch in stage_channels]

    return ESCNNFlexibleResNet(
        fields_per_stage=scaled_channels,
        blocks_per_stage=blocks_per_stage,
        num_classes=num_classes,
        in_channels=in_channels,
        act_cls=act_cls,
        rotations=rotations,
        reflection=reflection,
        max_frequency=max_frequency,
        stem_stride=stem_stride,
        pad_blocks=enable_auto_padding,
        pad_input=pad_input
    )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tet_conversion_parameter_counts():
    """Test that converted models have approximately the same parameter count."""
    print("\n" + "=" * 80)
    print("Testing parameter counts for model conversions")
    print("=" * 80)

    # Create base FlexibleResNet
    base_channels = [64, 128]
    blocks_per_stage = [2, 2]
    base_model = FlexibleResNet(
        channels=base_channels,
        blocks_per_stage=blocks_per_stage,
        num_classes=10,
        in_channels=1,
        activation=nn.GELU,
        stem_stride=1
    )

    base_params = count_parameters(base_model)
    print(f"\nBase FlexibleResNet parameters: {base_params:,}")

    # Test GroupResNet conversions
    print("\n" + "-" * 80)
    print("GroupResNet conversions:")
    print("-" * 80)

    for num_rotations in [4, 8]:
        for reflection in [False, True]:
            gcnn = convert_flexible_resnet_to_gcnn(
                base_model,
                num_rotations=num_rotations,
                reflection=reflection
            )
            gcnn_params = count_parameters(gcnn)
            ratio = gcnn_params / base_params
            group_type = "Dihedral" if reflection else "Cyclic"
            print(f"{group_type} N={num_rotations}: {gcnn_params:,} params (ratio: {ratio:.3f})")

    # Test ESCNN conversions
    print("\n" + "-" * 80)
    print("ESCNN conversions (discrete):")
    print("-" * 80)

    for rotations in [4, 8]:
        for reflection in [False, True]:
            escnn_model = convert_flexible_resnet_to_escnn(
                base_model,
                rotations=rotations,
                reflection=reflection,
                continuous=False
            )
            escnn_params = count_parameters(escnn_model)
            ratio = escnn_params / base_params
            group_type = "D" if reflection else "C"
            print(f"{group_type}{rotations}: {escnn_params:,} params (ratio: {ratio:.3f})")

    # Test ESCNN continuous conversions
    print("\n" + "-" * 80)
    print("ESCNN conversions (continuous):")
    print("-" * 80)

    for max_freq in [3, 5, 8]:
        for reflection in [False, True]:
            escnn_model = convert_flexible_resnet_to_escnn(
                base_model,
                rotations=8,  # Not used for continuous, but required
                reflection=reflection,
                continuous=True,
                max_frequency=max_freq
            )
            escnn_params = count_parameters(escnn_model)
            ratio = escnn_params / base_params
            group_type = "O(2)" if reflection else "SO(2)"
            print(f"{group_type} max_freq={max_freq}: {escnn_params:,} params (ratio: {ratio:.3f})")


if __name__ == "__main__":
    # Existing split tests
    model2 = FlexibleResNet(
        channels=[32, 64, 128, 256],
        blocks_per_stage=[2, 2, 2, 2],
        num_classes=10,
        in_channels=1,
        activation=nn.GELU,
        stem_stride=1
    )

    model_escnn = convert_flexible_resnet_to_escnn(model2).to("cuda")
    model_gcnn = convert_flexible_resnet_to_gcnn(model2).to("cuda")

    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(64, 1, 32, 32).to("cuda")
    y = torch.randint(0, 10, (64,)).to("cuda")
    model2.eval().to(device)
    start_time = time.time()
    for _ in range(1000):
        outputs = model2(x)
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    end_time = time.time()
    print(f"Average inference time over 100 runs: {(end_time - start_time) / 1000:.6f} seconds")

    x = x.to(device)
    y = y.to(device)
    model2.eval().to(device)
    start_time = time.time()
    for _ in range(1000):
        outputs = model2(x)
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    end_time = time.time()
    print(f"Average inference time over 100 runs: {(end_time - start_time) / 1000:.6f} seconds")
