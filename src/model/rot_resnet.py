import math



import torch
import torch.nn as nn
from escnn import nn as escnn_nn
from escnn import gspaces

class GBasicBlock(nn.Module):
    def __init__(self, in_type: escnn_nn.FieldType, out_fields: int, stride: int = 1,
                 act_cls=escnn_nn.ReLU, pad_blocks: bool = False):
        """
        Basic block for ESCNN ResNet.

        Args:
            in_type: Input field type.
            out_fields: Number of fields in the output (each field corresponds to one regular representation).
            stride: Stride for the first convolution in the block.
            act_cls: Activation class to use (e.g. escnn_nn.ReLU).
            pad_blocks: Whether to pad feature maps to odd dimensions before block convolutions.
        """
        super().__init__()

        gspace = in_type.gspace


        reg = gspace.regular_repr
        self.out_type = escnn_nn.FieldType(gspace, out_fields * [reg])
        is_continuous = False


        self.pad_blocks = pad_blocks

        # Convs
        self.conv1 = escnn_nn.R2Conv(in_type, self.out_type, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = escnn_nn.R2Conv(self.out_type, self.out_type, kernel_size=3, padding=1, bias=False)


        # Finite groups / regular repr: pointwise nonlinearities are allowed
        self.bn1 = escnn_nn.InnerBatchNorm(self.out_type)
        self.bn2 = escnn_nn.InnerBatchNorm(self.out_type)
        self.act = act_cls(self.out_type, inplace=True)

        # Projection for skip connection if needed
        self.proj = None
        if stride != 1 or in_type != self.out_type:
            self.proj = escnn_nn.R2Conv(in_type, self.out_type, kernel_size=1, stride=stride, bias=False)


    @staticmethod
    def pad_if_even(x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (1 - h % 2) % 2
        pad_w = (1 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x

    def forward(self, x: escnn_nn.GeometricTensor) -> escnn_nn.GeometricTensor:
        if self.pad_blocks:
            x = escnn_nn.GeometricTensor(self.pad_if_even(x.tensor), x.type)
        y = self.act(self.bn1(self.conv1(x)))
        if self.pad_blocks:
            y = escnn_nn.GeometricTensor(self.pad_if_even(y.tensor), y.type)
        y = self.bn2(self.conv2(y))
        skip = x if self.proj is None else self.proj(x)
        return self.act(y + skip)


class ESCNNFlexibleResNet(nn.Module):
    def __init__(self, fields_per_stage, blocks_per_stage, num_classes=10, in_channels=1,
                 act_cls=escnn_nn.ReLU, rotations=8, reflection=False, max_frequency=None, stem_stride=1, pad_blocks=False,
                 pad_input=False):
        super().__init__()
        """
        Flexible ESCNN ResNet architecture.

        Args:
            fields_per_stage: List of number of fields in each stage (e.g. [16, 32, 64]).
            blocks_per_stage: List of number of blocks in each stage (e.g. [2, 2, 2]).
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            act_cls: Activation class to use (e.g. escnn_nn.ReLU).
            rotations: Number of discrete rotations (N for C_N group).
            reflection: Whether to include reflections (D_N group) or not (C_N group).
            max_frequency: Max frequency for Fourier representations (not used here). Did not manage to get continuous working.
            stem_stride: Stride for the initial convolutional layer.
            pad_blocks: Whether to pad feature maps to odd dimensions before block convolutions.
            pad_input: Whether to pad input images to odd dimensions before processing.
        """


        if reflection:
            self.gspace = gspaces.flipRot2dOnR2(N=rotations)
        else:
            self.gspace = gspaces.rot2dOnR2(N=rotations)

        self.reflection = reflection
        self.max_frequency = max_frequency
        self.rotations = rotations
        self.pad_blocks = pad_blocks
        self.pad_input = pad_input


        first_type = escnn_nn.FieldType(self.gspace, fields_per_stage[0] * [self.gspace.regular_repr])
        in_type = escnn_nn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])

        # stem conv
        self.stem_conv = escnn_nn.R2Conv(in_type, first_type, kernel_size=3,
                                         stride=stem_stride, padding=1, bias=False)
        self.stem_stride = stem_stride

        # detect trivial irreps in first_type
        has_trivial = any(r.is_trivial() for r in first_type.representations)


        self.stem_bn = escnn_nn.InnerBatchNorm(first_type)
        self.stem_act = act_cls(first_type, inplace=True)

        # remaining architecture
        stage_modules = []
        current_type = first_type
        for stage_idx, (out_f, n_blocks) in enumerate(zip(fields_per_stage, blocks_per_stage)):
            blocks = []
            for block_idx in range(n_blocks):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                block = GBasicBlock(current_type, out_f, stride=stride, act_cls=act_cls, pad_blocks=pad_blocks)
                current_type = block.out_type
                blocks.append(block)
            stage_modules.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stage_modules)

        try:
            self.group_pool = escnn_nn.GroupPooling(current_type)
        except AssertionError:
            self.group_pool = escnn_nn.NormPool(current_type)

        self.fc = nn.Linear(fields_per_stage[-1], num_classes)

    @staticmethod
    def pad_to_odd_after_downsampling(x, num_downsamples):
        H, W = x.shape[-2:]
        factor = 2 ** num_downsamples
        H_pad = factor * ((H + factor - 1) // factor)
        if H_pad % 2 == 0:
            H_pad += 1
        pad_top = (H_pad - H) // 2
        pad_bottom = H_pad - H - pad_top
        W_pad = factor * ((W + factor - 1) // factor)
        if W_pad % 2 == 0:
            W_pad += 1
        pad_left = (W_pad - W) // 2
        pad_right = W_pad - W - pad_left
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    def forward(self, x):
        if self.pad_input:
            num_downsamples = len(self.stages) - 1
            if self.stem_stride > 1:
                num_downsamples += 1
            x = self.pad_to_odd_after_downsampling(x, num_downsamples)
        x = escnn_nn.GeometricTensor(x, self.stem_conv.in_type)
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        for stage in self.stages:
            x = stage(x)
        spatial_mean = x.tensor.mean(dim=(2, 3), keepdim=True)
        x = escnn_nn.GeometricTensor(spatial_mean, x.type)
        x = self.group_pool(x)
        x = x.tensor.view(x.tensor.size(0), -1)
        return self.fc(x)


import torch


import torch

@torch.no_grad()
def tst_ESCNN_invariance(model, N=4, device="cuda"):
    """
    Test invariance of the full ESCNN network (logits should stay the same
    under rotations by 0, 90, ..., (N-1)*360/N degrees).
    """
    model.eval()
    x = torch.randn(1, 1, 63, 63, device=device)

    # baseline output
    y0 = model(x)

    print("\n===== ESCNN FULL NETWORK INVARIANCE TEST =====")
    for k in range(N):
        xr = torch.rot90(x, k, dims=(-2, -1))
        yr = model(xr)

        # compare logits
        err = (yr - y0).abs().max().item()
        print(f"Rotation {k*90:3d}° | max difference = {err:.6f}")

    print("================================================\n")




if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import time

    # Example only – use your own ESCNN model
    model = ESCNNFlexibleResNet(
        fields_per_stage=[8, 16],
        blocks_per_stage=[2, 2],
        num_classes=10,
        in_channels=1,
        rotations=4,
        reflection=False,
        continuous=False,
        stem_stride=1
    ).cuda().eval()

    tst_ESCNN_invariance(model, N=4)

    from equiadapt import RotoReflectionEquivariantConv,RotationEquivariantConv

    block = GroupBasicBlock(
        in_ch=3, out_ch=6,
        num_rotations=4,
        use_reflection=False,
        stride=2
    ).cuda()

    errs = tst_equivariance(
        block,
        num_rotations=4,
        use_reflection=False
    )

    # Standard Conv2d baseline
    class StandardConv(nn.Module):
        def __init__(self, in_ch, out_ch, k, stride=1, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding)

        def forward(self, x):
            return self.conv(x)

    device = "cuda"

    # instantiate GroupResNet - keep group_pool='none' to preserve G axis
    net = GroupResNet(
        channels=[16],
        blocks_per_stage=[2],
        num_classes=10,
        in_channels=1,
        num_rotations=4,
        use_reflection=False,
        group_pool='none',   # preserve group axis for equivariance test
        pad_input=True,
        pad_blocks=False,
        stem_stride=1,
        stem_channels=16,
    ).to(device).eval()

    errs = tst_equivariance_resnet(
        net,
        num_rotations=4,
        use_reflection=False,
        B=2, in_ch=1, H=31, W=31, tol=1e-4, verbose=True
    )


    device = "cuda"
    B, Cin, Cout, H, W = 8, 64, 128, 64, 64
    num_rotations = 1

    x_equi = torch.randn(B, int(Cin/math.sqrt(1)), num_rotations, H, W, device=device)
    x = torch.randn(B, Cin*num_rotations, H, W, device=device)

    # Instantiate layers
    equiv = RotationEquivariantConv(int(Cin/math.sqrt(1)), int(Cout/math.sqrt(1)), 3, num_rotations=num_rotations, padding=1, device=device).eval().cuda()
    cnn = torch.nn.Conv2d(Cin  * num_rotations, Cout  * num_rotations, 3, padding=1).eval().cuda()

    # Warmup
    for _ in range(10):
        equiv(x_equi)
        cnn(x)



    torch.cuda.synchronize()
    t2 = time.time()
    for _ in range(100):
        cnn(x)
        torch.cuda.synchronize()
    t3 = time.time()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        equiv(x_equi)
        torch.cuda.synchronize()
    t1 = time.time()

    print(f"RotoReflectionConv: {(t1 - t0) / 50 * 1000:.2f} ms/iter")
    print(f"Standard CNN Conv  : {(t3 - t2) / 50 * 1000:.2f} ms/iter")
