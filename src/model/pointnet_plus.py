#taken from torchgeometric examples #TODO cite this
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, random_start=True):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)
        self.random_start = random_start

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio,random_start=self.random_start)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNetPlus(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # Replace MLP with standard Sequential for layer extraction
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x)

class PointNetPlusHalfSized(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Halve early layers
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 32, 32, 64]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([64 + 3, 64, 64, 128]))
        # Halve global aggregation MLP except keep output at 1024
        self.sa3_module = GlobalSAModule(MLP([128 + 3, 128, 256, 1024]))

        # Keep final MLP at full size
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x)


class PointNetPlusQuarterSized(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Quarter early layers
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 16, 16, 32]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([32 + 3, 32, 32, 64]))
        # Quarter global aggregation MLP except keep output at 1024
        self.sa3_module = GlobalSAModule(MLP([64 + 3, 64, 128, 1024]))

        # Keep final MLP at full size
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x)